import { useState, useRef, useCallback, useEffect } from "react";
import {
  ShieldCheck, Database, Cpu, Upload, FolderOpen, FileStack,
  CheckCircle2, XCircle, AlertTriangle, Loader2, ArrowRight,
  ArrowLeft, RotateCcw, Download, ChevronDown, ChevronUp, Camera, Square
} from "lucide-react";

const API_BASE = import.meta.env.VITE_TRUSTCV_API || "http://localhost:8000";

async function readJson(res) {
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

async function postForm(path, fields) {
  const form = new FormData();
  Object.entries(fields).forEach(([key, value]) => {
    if (value !== null && value !== undefined) form.append(key, value);
  });
  return readJson(await fetch(`${API_BASE}${path}`, { method: "POST", body: form }));
}

const VerificationAPI = {
  // These two endpoints intentionally keep Phase 3 and Phase 4 as separate requests.
  async phase3(model) {
    return postForm("/api/verify/model/phase3", {
      file: model,
      access_level: "white_box"
    });
  },
  async compatibility(model, dataset) {
    return postForm("/api/verify/dataset/compatibility", {
      file: model,
      reference_dataset: dataset
    });
  },
  async phase4(model, dataset, phase3Result) {
    return postForm("/api/verify/model/phase4", {
      file: model,
      reference_dataset: dataset,
      access_level: "white_box",
      phase3_result: JSON.stringify(phase3Result || {})
    });
  },
  async analyze(modelRef, inputFile, runId) {
    return postForm("/api/analyze", { file: inputFile, model_ref: modelRef, run_id: runId });
  },
  async analyzeBatch(modelRef, inputFiles, runId) {
    const form = new FormData();
    form.append("model_ref", modelRef);
    if (runId) form.append("run_id", runId);
    inputFiles.forEach(file => form.append("files", file));
    return readJson(await fetch(`${API_BASE}/api/analyze/batch`, { method: "POST", body: form }));
  },
  async runPdf(path, runId) {
    const form = new FormData();
    form.append("run_id", runId);
    const data = await readJson(await fetch(`${API_BASE}${path}`, { method: "POST", body: form }));
    const pdfRes = await fetch(`${API_BASE}/api/report/pdf/download/${encodeURIComponent(data.filename)}`);
    if (!pdfRes.ok) throw new Error("The PDF was generated but could not be downloaded.");
    return pdfRes.blob();
  }
};

const MODEL_STEPS = ["Model", "Phase 3", "Phase 4", "Input", "Analyze", "Result", "Provenance"];
const DATASET_STEPS = ["Select", "Verify", "Result"];

function StatusIcon({ state, size = 18 }) {
  if (state === "pass") return <CheckCircle2 size={size} color="#3DDC84" />;
  if (state === "fail") return <XCircle size={size} color="#FF6B6B" />;
  if (state === "warn") return <AlertTriangle size={size} color="#FFB454" />;
  if (state === "pending") return <Loader2 size={size} color="#8B93A3" className="tc-spin" />;
  return null;
}

function normalizeDisposition(value) {
  return String(value ?? "").trim().toLowerCase();
}

function checkState(c) {
  if (!c) return "pending";
  const disposition = normalizeDisposition(c.disposition || c.raw_disposition);
  if (["quarantine", "blocked", "reject", "rejected", "fail", "failed"].includes(disposition)) return "fail";
  if (["review", "warning", "warn"].includes(disposition)) return "warn";
  if (c.passed === null || c.passed === undefined) return "warn";
  return c.passed ? "pass" : "fail";
}

function inferDatasetRequirements(result) {
  const type = String(result?.model_type || "").toLowerCase();
  if (type === "smallcnn") return {
    recommended_dataset: "CIFAR-10 clean reference data",
    why: "The current SmallCNN integrity testbed expects clean 10-class CIFAR-10 reference images.",
    format: "Upload a CIFAR-10 binary ZIP or compatible clean classification dataset.",
    expected: { task: "image_classification", height: 32, width: 32, num_classes: 10 }
  };
  if (type === "yolo") return {
    recommended_dataset: "Clean object-detection reference images",
    why: "YOLO analysis needs clean images representative of the detector's deployed classes. COCO is appropriate only when the uploaded detector uses the COCO class space.",
    format: "Upload a ZIP, YOLO/COCO folder, or clean images. Labels and data.yaml are optional and strengthen the evidence when present.",
    expected: { task: "object_detection", num_classes: "Model-specific" }
  };
  return null;
}

function phase8Integrity(phase8) {
  return phase8?.integrity || phase8?.result || phase8?.evidence || phase8 || null;
}

function phase8Disposition(phase8) {
  const i = phase8Integrity(phase8);
  return normalizeDisposition(i?.disposition || i?.status);
}

function phase8State(phase8) {
  if (!phase8 || phase8.status === "placeholder") return "warn";
  const d = phase8Disposition(phase8);
  if (["quarantine", "blocked", "reject", "rejected", "fail", "failed", "flagged"].includes(d)) return "fail";
  if (["review", "warning", "warn"].includes(d)) return "warn";
  if (phase8.status === "real") return "pass";
  return "warn";
}

function phase6State(modelType, oodResult, shiftResult) {
  if (modelType === "smallcnn") return "warn";
  if (!oodResult) return "warn";
  if (oodResult.inDistribution === false || shiftResult?.shiftDetected) return "fail";
  if (oodResult.inDistribution === true) return "pass";
  return "warn";
}

function phase7State(modelType, phase7) {
  if (modelType === "smallcnn") return "warn";
  if (!phase7) return "warn";
  if (phase7.status !== "real") return "warn";
  const detections = Array.isArray(phase7.detections) ? phase7.detections : null;
  return detections && detections.length === 0 ? "warn" : "pass";
}

function phase7Detail(modelType, phase7) {
  if (modelType === "smallcnn") return "Round-1 placeholder";
  if (!phase7) return "Inference result not available";
  if (phase7.status !== "real") return phase7.detail || "Inference not executed";
  const n = Array.isArray(phase7.detections) ? phase7.detections.length : null;
  if (n === 0) return "Inference completed · 0 detections";
  if (n !== null) return `Inference completed · ${n} detection${n === 1 ? "" : "s"}`;
  return "Inference result available";
}

function phase8Detail(phase8) {
  if (!phase8) return "Integrity result not available";
  const i = phase8Integrity(phase8);
  const d = normalizeDisposition(i?.disposition || i?.status);
  if (["quarantine", "blocked", "reject", "rejected", "fail", "failed", "flagged"].includes(d)) {
    return `Integrity flagged · ${String(i?.disposition || i?.status).toUpperCase()}`;
  }
  return i?.reason || i?.detail || "Integrity result available";
}


function displayMetric(value, digits = 3) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number") return Number.isFinite(value) ? value.toFixed(digits) : "—";
  return String(value);
}

function percentMetric(value, digits = 1) {
  if (value === null || value === undefined || value === "") return "—";
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);
  return `${(n * 100).toFixed(digits)}%`;
}

function PhaseMetrics({ items }) {
  const visible = (items || []).filter(([, value]) => value !== undefined && value !== null && value !== "");
  if (!visible.length) return null;
  return (
    <div className="tc-phase-metrics">
      {visible.map(([label, value]) => (
        <div className="tc-phase-metric" key={label}>
          <span>{label}</span>
          <b>{String(value)}</b>
        </div>
      ))}
    </div>
  );
}

function PhaseInfo({ phase, result, modelType, cameraCount = 0 }) {
  if (!result && !phase) return null;
  if (phase === 3) {
    const mirad = result?.mirad || result?.artifact_identity || {};
    const identity = mirad?.evidence?.candidate || mirad?.candidate || {};
    return <PhaseMetrics items={[
      ["Model type", result?.model_type || modelType],
      ["Checks", Array.isArray(result?.checks) ? result.checks.length : undefined],
      ["Passed checks", Array.isArray(result?.checks) ? result.checks.filter(c => c?.passed === true).length : undefined],
      ["SHA-256", result?.sha256 ? `${String(result.sha256).slice(0, 12)}…` : undefined],
      ["Artifact", identity?.artifact_id || mirad?.artifact_id],
      ["Version", identity?.version || mirad?.version],
    ]} />;
  }
  if (phase === 4) {
    const m = result?.dataset_metrics || {};
    const c = result?.calibration || {};
    const demo = result?.demo_mode || {};
    return <PhaseMetrics items={[
      ["Method", result?.method],
      ["Mean TRACE", displayMetric(m.mean_trace_score, 4)],
      ["P90 TRACE", displayMetric(m.p90_trace_score, 4)],
      ["Threshold", displayMetric(m.threshold, 4)],
      ["Suspicious", percentMetric(m.suspicious_fraction)],
      ["Calibration", c?.used ? `${c.reference_images ?? "—"} images` : "Not calibrated"],
      ["Demo split", demo?.enabled ? `${demo.calibration_images} + ${demo.evaluation_images}` : undefined],
      ["CTC / FTC", demo?.enabled ? `${demo.ctc_samples} / ${demo.ftc_samples}` : undefined],
    ]} />;
  }
  if (phase === 6) {
    const o = result || {};
    const s = o?.shift || {};
    return <PhaseMetrics items={[
      ["In distribution", o?.inDistribution === true ? "YES" : o?.inDistribution === false ? "NO" : undefined],
      ["OOD score", displayMetric(o?.confidence, 4)],
      ["Shift", s?.shiftDetected === true ? "DETECTED" : s?.shiftDetected === false ? "NONE" : undefined],
      ["Severity", s?.severity],
      ["Reasons", Array.isArray(o?.reasons) ? o.reasons.length : undefined],
      ["Input source", cameraCount ? `Camera · ${cameraCount} frames` : undefined],
    ]} />;
  }
  if (phase === 7) {
    const ds = Array.isArray(result?.detections) ? result.detections : [];
    const confs = ds.map(d => Number(d?.confidence)).filter(Number.isFinite);
    const mean = confs.length ? confs.reduce((a,b)=>a+b,0)/confs.length : null;
    return <PhaseMetrics items={[
      ["Detections", ds.length],
      ["Mean confidence", percentMetric(mean)],
      ["Max confidence", percentMetric(confs.length ? Math.max(...confs) : null)],
      ["Min confidence", percentMetric(confs.length ? Math.min(...confs) : null)],
      ["Confidence threshold", result?.conf_threshold ?? result?.confidence_threshold],
      ["IoU threshold", result?.iou_threshold],
      ["Precision", result?.precision ?? result?.metrics?.precision],
      ["Recall", result?.recall ?? result?.metrics?.recall],
      ["F1", result?.f1 ?? result?.metrics?.f1],
      ["mAP50", result?.map50 ?? result?.mAP50 ?? result?.metrics?.map50],
    ]} />;
  }
  if (phase === 8) {
    const i = phase8Integrity(result) || {};
    const metrics = result?.metrics || i?.metrics || {};
    const robustness = result?.robustness || i?.robustness || metrics?.robustness || i?.robustness_metrics || {};
    const confidence = result?.confidence || i?.confidence || metrics?.confidence || i?.confidence_metrics || {};
    const calibration = result?.calibration || i?.calibration || metrics?.calibration || {};
    const accuracy = result?.accuracy || i?.accuracy || metrics?.accuracy || {};
    return <PhaseMetrics items={[
      ["Disposition", i?.disposition || i?.status],
      ["Integrity confidence", percentMetric(i?.confidence)],
      ["Robustness", robustness?.score !== undefined ? displayMetric(robustness.score, 4) : robustness?.robustness_score !== undefined ? displayMetric(robustness.robustness_score, 4) : undefined],
      ["Calibration", calibration?.score !== undefined ? displayMetric(calibration.score, 4) : calibration?.status],
      ["Confidence stability", confidence?.stability !== undefined ? displayMetric(confidence.stability, 4) : confidence?.std !== undefined ? displayMetric(confidence.std, 4) : undefined],
      ["Reliability", i?.reliability_score !== undefined ? displayMetric(i.reliability_score, 4) : metrics?.reliability_score !== undefined ? displayMetric(metrics.reliability_score, 4) : undefined],
      ["Accuracy", accuracy?.value !== undefined && accuracy?.value !== null ? percentMetric(accuracy.value) : accuracy?.status === "not_computable" ? "N/A" : undefined],
      ["Calibration", calibration?.score !== undefined && calibration?.score !== null ? displayMetric(calibration.score, 4) : calibration?.status === "not_computable" ? "N/A" : undefined],
      ["Input hash", i?.input_hash ? `${String(i.input_hash).slice(0, 12)}…` : undefined],
      ["Output hash", i?.output_hash ? `${String(i.output_hash).slice(0, 12)}…` : undefined],
    ]} />;
  }
  if (phase === 9) {
    return <PhaseMetrics items={[
      ["Record hash", result?.record_hash ? `${String(result.record_hash).slice(0, 16)}…` : undefined],
      ["Status", result?.status],
      ["Events recorded", result?.events_recorded],
      ["Ledger", result?.ledger_verification === true || result?.ledger_valid === true ? "VALID" : result?.ledger_verification === false || result?.ledger_valid === false ? "INVALID" : result?.ledger_message || undefined],
      ["Model SHA-256", result?.model_sha256 ? `${String(result.model_sha256).slice(0, 12)}…` : undefined],
      ["Frames audited", cameraCount || undefined],
      ["Timestamp", result?.timestamp_utc || result?.timestamp],
    ]} />;
  }
  return null;
}

function HashCheckpoint({ checkpoint }) {
  if (!checkpoint) return null;
  const verified = checkpoint.verified === true;
  return (
    <div className={`tc-hash-checkpoint ${verified ? "pass" : "fail"}`}>
      <StatusIcon state={verified ? "pass" : "fail"} size={17}/>
      <div className="tc-hash-checkpoint-main">
        <b>{verified ? "HASH RE-VERIFIED" : "MODEL HASH VERIFICATION FAILED"}</b>
        <span>{verified ? "NO MODEL MODIFICATION DETECTED" : "MODEL MODIFICATION / SUBSTITUTION DETECTED"}</span>
      </div>
      <div className="tc-hash-checkpoint-digest">{String(checkpoint.actual_model_sha256 || checkpoint.model_sha256 || "").slice(0, 16)}…</div>
    </div>
  );
}

function RunPdfButton({ runId, endpoint, filename, label }) {
  const [busy, setBusy] = useState(false);
  const save = async () => {
    if (!runId) return;
    setBusy(true);
    try {
      const blob = await VerificationAPI.runPdf(endpoint, runId);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a"); a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url);
    } catch (e) { window.alert(e.message || "PDF could not be generated."); }
    finally { setBusy(false); }
  };
  return <button type="button" className="tc-secondary-btn" onClick={save} disabled={!runId || busy}><Download size={15}/>{busy ? "Generating…" : label}</button>;
}

function CheckCard({ label, state, detail, evidence }) {
  const [open, setOpen] = useState(false);
  const hasEvidence = Boolean(detail || (evidence && Object.keys(evidence).length));
  return (
    <div className={`tc-checkcard ${open ? "open" : ""}`}>
      <button
        type="button"
        className="tc-checkcard-button"
        disabled={!hasEvidence}
        onClick={() => hasEvidence && setOpen(v => !v)}
      >
        <StatusIcon state={state} />
        <span className="tc-checkcard-main">
          <span className="tc-checkcard-label">{label}</span>
          {detail && !open && <span className="tc-checkcard-summary">{detail}</span>}
        </span>
        {hasEvidence && (open ? <ChevronUp size={16} /> : <ChevronDown size={16} />)}
      </button>
      {open && hasEvidence && (
        <div className="tc-checkcard-detail">
          {detail && <div className="tc-evidence-note">{detail}</div>}
          {Object.entries(evidence || {}).map(([key, value]) => (
            <div className="tc-evidence-row" key={key}>
              <span>{key.replaceAll("_", " ")}</span>
              <strong>{typeof value === "object" ? JSON.stringify(value, null, 2) : String(value)}</strong>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function PhaseCard({ phase, title, state, detail, children }) {
  const [open, setOpen] = useState(false);
  return (
    <div className={`tc-phase-card ${open ? "open" : ""}`}>
      <button type="button" className="tc-phase-card-button" onClick={() => setOpen(v => !v)}>
        <div>
          <div className="tc-phase-card-phase">{phase}</div>
          <div className="tc-phase-card-title">{title}</div>
          {detail && <div className="tc-phase-card-detail">{detail}</div>}
        </div>
        <div className="tc-phase-card-right">
          <StatusIcon state={state} />
          {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </div>
      </button>
      {open && <div className="tc-phase-card-body">{children}</div>}
    </div>
  );
}

function Rail({ steps, current }) {
  return (
    <div className="tc-rail">
      {steps.map((label, i) => {
        const n = i + 1;
        const done = n < current;
        const active = n === current;
        return (
          <div className="tc-rail-item" key={label}>
            <div className={`tc-rail-dot ${done ? "done" : ""} ${active ? "active" : ""}`}>
              {done ? <CheckCircle2 size={14} /> : n}
            </div>
            <span className={`tc-rail-label ${active ? "active" : ""}`}>{label}</span>
            {i < steps.length - 1 && <div className={`tc-rail-line ${done ? "done" : ""}`} />}
          </div>
        );
      })}
    </div>
  );
}

function Panel({ children }) { return <div className="tc-panel">{children}</div>; }

function Header({ number, eyebrow, title, subtitle }) {
  return (
    <div className="tc-phase-header">
      <div className="tc-phase-number">{String(number).padStart(2, "0")}</div>
      <div>
        <div className="tc-eyebrow">{eyebrow}</div>
        <h1 className="tc-title">{title}</h1>
        {subtitle && <p className="tc-lede">{subtitle}</p>}
      </div>
    </div>
  );
}

function Banner({ status, text }) {
  const m = {
    trusted: ["#3DDC84", "rgba(61,220,132,.07)", <ShieldCheck size={20} />, "ACCEPTED"],
    risk: ["#FF6B6B", "rgba(255,107,107,.07)", <XCircle size={20} />, "QUARANTINED"],
    warn: ["#FFB454", "rgba(255,180,84,.07)", <AlertTriangle size={20} />, "REVIEW"]
  }[status] || ["#FFB454", "rgba(255,180,84,.07)", <AlertTriangle size={20} />, "REVIEW"];
  return (
    <div className="tc-banner" style={{ borderColor: m[0], background: m[1] }}>
      <div className="tc-banner-top" style={{ color: m[0] }}>{m[2]}<span>{m[3]}</span></div>
      <p>{text}</p>
    </div>
  );
}

function UploadField({ label, sub, accept, value, onFile }) {
  const ref = useRef(null);
  return (
    <div className="tc-upload-field" onClick={() => ref.current?.click()}>
      <div className="tc-upload-icon"><Upload size={20} /></div>
      <div className="tc-upload-copy">
        <div className="tc-upload-label">{label}</div>
        <div className="tc-upload-sub">{value ? value.name : sub}</div>
      </div>
      <span className="tc-upload-action">{value ? "Selected" : "Choose file"}</span>
      <input ref={ref} className="tc-hidden-input" type="file" accept={accept} onChange={e => e.target.files?.[0] && onFile(e.target.files[0])} />
    </div>
  );
}

function crc32(bytes) {
  let crc = 0xFFFFFFFF;
  for (let i = 0; i < bytes.length; i++) {
    crc ^= bytes[i];
    for (let j = 0; j < 8; j++) {
      crc = (crc >>> 1) ^ (0xEDB88320 & -(crc & 1));
    }
  }
  return (crc ^ 0xFFFFFFFF) >>> 0;
}

function writeU16(view, offset, value) { view.setUint16(offset, value, true); }
function writeU32(view, offset, value) { view.setUint32(offset, value >>> 0, true); }

async function filesToZip(fileList) {
  const files = Array.from(fileList || []).filter(f => {
    const n = f.name.toLowerCase();
    return f.size >= 0 && !n.endsWith('.cache') && !n.endsWith('.ds_store') && !n.startsWith('__macosx');
  });
  if (!files.length) throw new Error("No usable files were selected.");

  const rawPaths = files.map(f => f.webkitRelativePath || f.name);
  const firstParts = rawPaths.map(p => p.split('/')[0]);
  const stripRoot = rawPaths.every(p => p.includes('/')) && new Set(firstParts).size === 1;
  const entries = [];

  for (let i = 0; i < files.length; i++) {
    const originalPath = rawPaths[i].replaceAll('\\', '/');
    let path = stripRoot ? originalPath.split('/').slice(1).join('/') : originalPath;
    path = path.replace(/^\/+/, '');
    if (!path || path.endsWith('/')) continue;
    const data = new Uint8Array(await files[i].arrayBuffer());
    const nameBytes = new TextEncoder().encode(path);
    entries.push({ path, data, nameBytes, crc: crc32(data) });
  }

  if (!entries.length) throw new Error("The selected folder contains no usable files.");

  const chunks = [];
  const central = [];
  let offset = 0;
  for (const e of entries) {
    const local = new ArrayBuffer(30 + e.nameBytes.length);
    const v = new DataView(local);
    writeU32(v, 0, 0x04034b50);
    writeU16(v, 4, 20);
    writeU16(v, 6, 0x0800);
    writeU16(v, 8, 0);
    writeU16(v, 10, 0);
    writeU16(v, 12, 0);
    writeU32(v, 14, e.crc);
    writeU32(v, 18, e.data.length);
    writeU32(v, 22, e.data.length);
    writeU16(v, 26, e.nameBytes.length);
    writeU16(v, 28, 0);
    new Uint8Array(local, 30).set(e.nameBytes);
    chunks.push(local, e.data);

    const c = new ArrayBuffer(46 + e.nameBytes.length);
    const cv = new DataView(c);
    writeU32(cv, 0, 0x02014b50);
    writeU16(cv, 4, 20);
    writeU16(cv, 6, 20);
    writeU16(cv, 8, 0x0800);
    writeU16(cv, 10, 0);
    writeU16(cv, 12, 0);
    writeU16(cv, 14, 0);
    writeU32(cv, 16, e.crc);
    writeU32(cv, 20, e.data.length);
    writeU32(cv, 24, e.data.length);
    writeU16(cv, 28, e.nameBytes.length);
    writeU16(cv, 30, 0);
    writeU16(cv, 32, 0);
    writeU16(cv, 34, 0);
    writeU16(cv, 36, 0);
    writeU32(cv, 38, 0);
    writeU32(cv, 42, offset);
    new Uint8Array(c, 46).set(e.nameBytes);
    central.push(c);

    offset += local.byteLength + e.data.length;
  }

  const centralOffset = offset;
  let centralSize = 0;
  for (const c of central) { chunks.push(c); centralSize += c.byteLength; }

  const end = new ArrayBuffer(22);
  const ev = new DataView(end);
  writeU32(ev, 0, 0x06054b50);
  writeU16(ev, 4, 0);
  writeU16(ev, 6, 0);
  writeU16(ev, 8, entries.length);
  writeU16(ev, 10, entries.length);
  writeU32(ev, 12, centralSize);
  writeU32(ev, 16, centralOffset);
  writeU16(ev, 20, 0);
  chunks.push(end);

  return new File(chunks, `reference_dataset_${Date.now()}.zip`, { type: 'application/zip' });
}

function DatasetUploadField({ value, onDataset }) {
  const zipRef = useRef(null);
  const folderRef = useRef(null);
  const imagesRef = useRef(null);
  const [busy, setBusy] = useState(false);

  const makeDataset = async (files) => {
    if (!files?.length) return;
    setBusy(true);
    try {
      const first = files[0];
      const isZip = first.name.toLowerCase().endsWith('.zip') && files.length === 1;
      const dataset = isZip ? first : await filesToZip(files);
      onDataset(dataset);
    } catch (e) {
      window.alert(e.message || 'Could not prepare the selected reference data.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="tc-dataset-picker">
      <div className="tc-dataset-picker-title"><Database size={17}/><div><b>{value ? 'Reference data selected' : 'Choose clean reference data'}</b><span>{value ? 'Reference dataset prepared for compatibility and anomaly analysis' : 'ZIP, YOLO/COCO folder, or multiple clean images'}</span></div></div>
      <div className="tc-dataset-picker-actions">
        <button type="button" className="tc-source-card" onClick={() => zipRef.current?.click()} disabled={busy}>
          <Upload size={20}/><span>Upload ZIP</span><span className="tc-source-sub">dataset archive</span>
        </button>
        <button type="button" className="tc-source-card" onClick={() => folderRef.current?.click()} disabled={busy}>
          <FolderOpen size={20}/><span>Select Folder</span><span className="tc-source-sub">YOLO / COCO / images</span>
        </button>
        <button type="button" className="tc-source-card" onClick={() => imagesRef.current?.click()} disabled={busy}>
          <FileStack size={20}/><span>Select Images</span><span className="tc-source-sub">one or many clean images</span>
        </button>
      </div>
      <input ref={zipRef} className="tc-hidden-input" type="file" accept=".zip,application/zip" onChange={e => makeDataset(e.target.files)}/>
      <input ref={folderRef} className="tc-hidden-input" type="file" webkitdirectory="" directory="" multiple onChange={e => makeDataset(e.target.files)}/>
      <input ref={imagesRef} className="tc-hidden-input" type="file" accept="image/*" multiple onChange={e => makeDataset(e.target.files)}/>
      {busy && <div className="tc-compatibility-box warn"><Loader2 size={16} className="tc-spin"/><span>Preparing selected reference data…</span></div>}
      {value && !busy && <div className="tc-dataset-selected">Selected reference data · quick demo uses 15 calibration + 7 evaluation images</div>}
    </div>
  );
}

function SourcePicker({ onUpload, onExisting, existingLabel, existingIcon }) {
  const ref = useRef(null);
  return (
    <div className="tc-source-grid">
      <button type="button" className="tc-source-card" onClick={() => ref.current?.click()}>
        <Upload size={22} /><span>Upload</span><span className="tc-source-sub">from your device</span>
      </button>
      <input ref={ref} className="tc-hidden-input" type="file" onChange={e => e.target.files?.[0] && onUpload(e.target.files[0])} />
      <button type="button" className="tc-source-card" onClick={onExisting}>
        {existingIcon}<span>{existingLabel}</span><span className="tc-source-sub">already registered</span>
      </button>
    </div>
  );
}

function Loading({ phase, modelName }) {
  return (
    <div className="tc-loading">
      <Loader2 size={34} className="tc-spin tc-loader" />
      <div className="tc-loading-title">{phase === 4 ? "Analyzing model integrity" : "Establishing model identity"}</div>
      <div className="tc-loading-model">{modelName}</div>
      <div className="tc-progress"><div /></div>
      {phase === 4 && <div className="tc-loading-stages"><span>Neural Cleanse</span><span>Backdoor analysis</span><span>Ablation validation</span><span>STRIP</span></div>}
      <p>{phase === 4 ? "This analysis may take a few minutes. The result will appear here when the backend finishes." : "Running the model identity and trust checks."}</p>
    </div>
  );
}

function DownloadButton({ filename, payload, label, pdf = false }) {
  const [downloading, setDownloading] = useState(false);

  const save = async () => {
    if (!pdf) {
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      return;
    }

    setDownloading(true);
    try {
      const form = new FormData();
      form.append("report", JSON.stringify(payload));

      const res = await fetch(`${API_BASE}/api/report/pdf`, {
        method: "POST",
        body: form
      });

      const data = await res.json().catch(() => ({}));
      if (!res.ok || data.error) {
        throw new Error(data.error || `HTTP ${res.status}`);
      }

      const pdfRes = await fetch(
        `${API_BASE}/api/report/pdf/download/${encodeURIComponent(data.filename)}`
      );

      if (!pdfRes.ok) {
        const downloadError = await pdfRes.json().catch(() => ({}));
        throw new Error(downloadError.error || "The PDF was generated but could not be downloaded.");
      }

      const blob = await pdfRes.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename.endsWith(".pdf") ? filename : `${filename}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      window.alert(e.message || "PDF report could not be generated.");
    } finally {
      setDownloading(false);
    }
  };

  return (
    <button
      type="button"
      className="tc-secondary-btn"
      onClick={save}
      disabled={downloading}
    >
      {downloading ? <Loader2 size={15} className="tc-spin" /> : <Download size={15} />}
      {downloading ? "Generating PDF..." : label}
    </button>
  );
}

function Footer({ reset }) {
  return <div className="tc-footer"><button type="button" className="tc-ghost-btn" onClick={reset}><RotateCcw size={15} /> Run another verification</button></div>;
}


function CameraCapture({ onComplete, disabled = false }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const timerRef = useRef(null);
  const framesRef = useRef([]);
  const [active, setActive] = useState(false);
  const [count, setCount] = useState(0);
  const [cameraError, setCameraError] = useState(null);

  const stopStream = useCallback(() => {
    if (timerRef.current) { window.clearInterval(timerRef.current); timerRef.current = null; }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setActive(false);
  }, []);

  useEffect(() => () => stopStream(), [stopStream]);

  const captureFrame = useCallback(() => {
    const video = videoRef.current, canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2 || !video.videoWidth) return;
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(blob => {
      if (!blob) return;
      const file = new File([blob], `camera_frame_${Date.now()}.jpg`, {type:"image/jpeg"});
      file.previewUrl = URL.createObjectURL(blob);
      const next = [...framesRef.current, file].slice(-9);
      framesRef.current.filter(x => !next.includes(x) && x.previewUrl).forEach(x => URL.revokeObjectURL(x.previewUrl));
      framesRef.current = next;
      setCount(next.length);
    }, "image/jpeg", 0.88);
  }, []);

  const startCamera = async () => {
    if (disabled || active) return;
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {facingMode:"environment", width:{ideal:1280}, height:{ideal:720}},
        audio:false
      });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      framesRef.current = []; setCount(0); setActive(true);
      captureFrame();
      timerRef.current = window.setInterval(captureFrame, 750);
    } catch (e) {
      setCameraError(e?.name === "NotAllowedError"
        ? "Camera permission was denied. Allow camera access and try again."
        : "Could not start the camera. Check that a camera is available.");
    }
  };

  const stopCamera = () => {
    if (!active) return;
    if (timerRef.current) { window.clearInterval(timerRef.current); timerRef.current = null; }
    captureFrame();
    window.setTimeout(() => {
      const frames = framesRef.current.slice(-9);
      stopStream();
      if (frames.length < 6) {
        setCameraError(`Capture at least 6 frames before stopping. Currently captured ${frames.length}.`);
        return;
      }
      onComplete(frames);
    }, 150);
  };

  return <div className="tc-camera-box">
    <div className="tc-camera-preview">
      <video ref={videoRef} className="tc-camera-video" muted playsInline />
      {!active && <div className="tc-camera-placeholder"><Camera size={28}/><span>Camera is off</span><small>Use Start Camera to begin</small></div>}
      <canvas ref={canvasRef} className="tc-hidden-canvas" />
    </div>
    <div className="tc-camera-controls">
      {!active
        ? <button type="button" className="tc-primary-btn" disabled={disabled} onClick={startCamera}><Camera size={16}/> Start Camera</button>
        : <button type="button" className="tc-primary-btn" onClick={stopCamera}><Square size={15}/> Stop & Analyze</button>}
      <div className="tc-camera-count"><span>CAPTURED</span><b>{count}/9</b></div>
    </div>
    <div className="tc-camera-help">Frames are sampled during the live session. Stop sends the latest 6–9 frames through the same Phase 6–9 assurance path as uploaded images.</div>
    {cameraError && <div className="tc-error">{cameraError}</div>}
  </div>;
}

export default function TrustCV() {
  const [mode, setMode] = useState(null);
  const [step, setStep] = useState(0);
  const [busy, setBusy] = useState(false);
  const [modelFile, setModelFile] = useState(null);
  const [referenceDataset, setReferenceDataset] = useState(null);
  const [inputName, setInputName] = useState("");
  const [phase3Result, setPhase3Result] = useState(null);
  const [phase4Result, setPhase4Result] = useState(null);
  const [datasetRequirements, setDatasetRequirements] = useState(null);
  const [datasetResult, setDatasetResult] = useState(null);
  const [compatibilityResult, setCompatibilityResult] = useState(null);
  const [compatibilityBusy, setCompatibilityBusy] = useState(false);
  const [oodResult, setOodResult] = useState(null);
  const [shiftResult, setShiftResult] = useState(null);
  const [oodInputSource, setOodInputSource] = useState(null);
  const [cameraFrames, setCameraFrames] = useState([]);
  const [cameraResults, setCameraResults] = useState([]);
  const [error, setError] = useState(null);

  const reset = useCallback(() => {
    setMode(null); setStep(0); setBusy(false); setModelFile(null); setReferenceDataset(null);
    setInputName(""); setPhase3Result(null); setPhase4Result(null); setDatasetRequirements(null); setDatasetResult(null); setCompatibilityResult(null); setCompatibilityBusy(false);
    setOodResult(null); setShiftResult(null); setOodInputSource(null); setCameraFrames([]); setCameraResults([]); setError(null);
  }, []);

  const phase3Checks = phase3Result?.phase3?.checks || phase3Result?.checks || phase3Result?.mirad?.checks || [];
  const phase4 = phase4Result?.phase4 || null;
  const phase4Flags = phase4?.flags || [];
  const phase3Passed = phase3Result?.passed === true;
  const phase3Review = phase3Result?.status === "review" || phase3Result?.disposition === "review" || phase3Checks.some(c => c?.passed === null || c?.passed === undefined);
  const phase4Disposition = normalizeDisposition(phase4?.disposition);
  const phase4Accepted = phase4Disposition === "accept";
  const phase4Demo = phase4?.demo_mode || null;
  const phase4Metrics = phase4?.dataset_metrics || {};
  const phase4Calibration = phase4?.calibration || {};
  const phase4Suspicious = Number(phase4Metrics.suspicious_fraction || 0);
  const phase4Quarantined = ["quarantine", "blocked", "reject", "rejected", "fail", "failed"].includes(phase4Disposition);
  const yoloDemoContinue =
    phase4Result?.model_type === "yolo" &&
    phase4?.status === "placeholder";
  const canContinueToInference = phase4Accepted || yoloDemoContinue;

  const runPhase3 = async () => {
    if (!modelFile) return;
    setInputName(modelFile.name); setStep(2); setBusy(true); setError(null);
    try {
      const result = await VerificationAPI.phase3(modelFile);
      setPhase3Result(result);
      setDatasetRequirements(result.dataset_requirements || inferDatasetRequirements(result));
    } catch (e) {
      setError(e.message || "Phase 3 verification could not complete.");
      setPhase3Result(null);
    } finally { setBusy(false); }
  };

  const handleReferenceDataset = async (dataset) => {
    setReferenceDataset(dataset);
    setCompatibilityResult(null);
    setError(null);
    if (!modelFile || !phase3Result || !dataset) return;
    setCompatibilityBusy(true);
    try {
      const result = await VerificationAPI.compatibility(modelFile, dataset);
      setCompatibilityResult(result);
    } catch (e) {
      setCompatibilityResult({
        compatible: false,
        disposition: "blocked",
        error: e.message || "Dataset compatibility check failed."
      });
    } finally {
      setCompatibilityBusy(false);
    }
  };

  const runPhase4 = async () => {
    if (!modelFile || !referenceDataset || !phase3Result) return;
    const compatibility = compatibilityResult?.compatibility || compatibilityResult;
    const compatible = compatibility?.compatible ?? compatibilityResult?.passed ?? false;
    if (!compatible) {
      setError(compatibility?.errors?.[0] || compatibilityResult?.error || "Reference dataset is not compatible with this model.");
      return;
    }
    setStep(3); setBusy(true); setError(null);
    try {
      const result = await VerificationAPI.phase4(modelFile, referenceDataset, phase3Result);
      setPhase4Result(result);
    } catch (e) {
      setError(e.message || "Phase 4 integrity analysis could not complete.");
      setPhase4Result(null);
    } finally { setBusy(false); }
  };

  const runAnalysis = async (source, inputFile = null) => {
    setOodInputSource(source); setStep(5); setBusy(true); setError(null);
    try {
      if (!canContinueToInference) throw new Error("Phase 4 must accept the model before inference testing.");
      if (!phase4Result?.model_ref) throw new Error("No verified model reference is available.");
      if (phase4Result.model_type !== "yolo") {
        const p = phase4Result.phase6_9;
        setOodResult(p?.phase6 || null);
        setShiftResult({shiftDetected:null, severity:"none", detail:"Phase 6-9 are explicit placeholders for SmallCNN in Round 1."});
        setStep(6); return;
      }
      if (!inputFile) throw new Error("Please upload a test image for the YOLO analysis path.");
      const result = await VerificationAPI.analyze(phase4Result.model_ref, inputFile, phase4Result.run_id);
      setCameraResults([]);
      setOodResult(result.phase6 || null);
      setShiftResult({
        shiftDetected: result.phase6?.inDistribution === false,
        severity: result.phase6?.inDistribution === false ? "medium" : "none",
        detail: result.phase6?.detail || "Phase 6 completed."
      });
      setPhase4Result(prev => ({...prev, analysis:result}));
      setStep(6);
    } catch (e) {
      setError(e.message || "Testing could not complete."); setStep(4);
    } finally { setBusy(false); }
  };

  const runCameraAnalysis = async (frames) => {
    if (!frames || frames.length < 6 || frames.length > 9) {
      setError("Live camera analysis requires 6 to 9 captured frames."); return;
    }
    setCameraFrames(frames); setCameraResults([]); setOodInputSource("camera");
    setInputName(`Live camera · ${frames.length} frames`); setStep(5); setBusy(true); setError(null);
    try {
      if (!canContinueToInference) throw new Error("Phase 4 must accept the model before inference testing.");
      if (!phase4Result?.model_ref) throw new Error("No verified model reference is available.");
      const result = await VerificationAPI.analyzeBatch(phase4Result.model_ref, frames, phase4Result.run_id);
      setCameraResults(result.frames || []);
      setOodResult(result.phase6 || null);
      setShiftResult({
        shiftDetected: result.phase6?.inDistribution === false,
        severity: result.phase6?.inDistribution === false ? "medium" : "none",
        detail: result.phase6?.detail || "Camera batch completed."
      });
      setPhase4Result(prev => ({...prev, analysis:result}));
      setStep(6);
    } catch (e) {
      setError(e.message || "Camera analysis could not complete."); setStep(4);
    } finally { setBusy(false); }
  };

  const phase3Report = {
    report_type: "TrustCV Phase 3 — Trust & Identity Report",
    report_scope: "phase3",
    generated_at: new Date().toISOString(),
    model: { filename: inputName, model_type: phase3Result?.model_type, sha256: phase3Result?.sha256 },
    mirad: phase3Result?.mirad || phase3Result?.artifact_identity || null,
    provenance: phase3Result?.provenance || null,
    checks: phase3Checks,
    decision: phase3Passed ? "accepted_for_phase4" : "stopped"
  };

  const phase4Report = {
    report_type: "TrustCV Phase 4 — Model Integrity Report",
    report_scope: "phase4",
    generated_at: new Date().toISOString(),
    model: { filename: inputName, model_type: phase4Result?.model_type, sha256: phase4Result?.sha256 },
    integrity: phase4,
    flags: phase4Flags,
    decision: phase4?.disposition || null
  };

  const phase8OverallDisposition = phase8Disposition(phase4Result?.analysis?.phase8);
  const phase6OverallState = phase6State(phase4Result?.model_type, oodResult, shiftResult);
  const overallDisposition =
    phase4Quarantined || ["quarantine", "blocked", "reject", "rejected", "fail", "failed"].includes(phase8OverallDisposition)
      ? "quarantine"
      : phase6OverallState === "fail"
        ? "quarantine"
        : phase4Result?.model_type === "smallcnn"
          ? "review"
          : "accept";

  const analysisResult = phase4Result?.analysis || {};
  // /api/analyze returns one provenance object; /api/analyze/batch returns an array.
  // Normalize both shapes so the Provenance page always renders the actual records.
  const rawProvenance = analysisResult?.provenance;
  const provenanceRecords = Array.isArray(rawProvenance)
    ? rawProvenance
    : rawProvenance
      ? [rawProvenance]
      : [];
  const anomalyFindings = Array.isArray(analysisResult?.findings)
    ? analysisResult.findings
    : analysisResult?.findings
      ? [analysisResult.findings]
      : [];
  const auditSummary = analysisResult?.phase9 || {};

  const completeReport = {
    report_type: "TrustCV Complete Inference Report",
    report_scope: "complete",
    generated_at: new Date().toISOString(),
    run_id: phase4Result?.run_id || analysisResult?.run_id || null,
    overall_disposition: overallDisposition,
    overall_status: overallDisposition === "quarantine" ? "quarantined" : overallDisposition === "review" ? "review" : "accepted",
    model: {
      filename: inputName,
      model_type: phase4Result?.model_type,
      sha256: phase4Result?.sha256
    },
    access_level: phase4?.access_level || "white_box",
    reference_data: phase4Result?.reference_data || phase3Result?.reference_data || null,
    phase3: phase3Report,
    phase4: phase4Report,
    phase6: oodResult,
    phase7: phase4Result?.analysis?.phase7 || null,
    phase8: phase4Result?.analysis?.phase8 || null,
    phase9: phase4Result?.analysis?.phase9 || null,
    shift: shiftResult,
    record_hash: phase4Result?.analysis?.record_hash || null,
    inference: {
      input_source: oodInputSource || null,
      frame_count: oodInputSource === "camera" ? cameraResults.length : 1,
      camera: oodInputSource === "camera",
      record_hash: phase4Result?.analysis?.record_hash || null,
      record_hashes: phase4Result?.analysis?.phase9?.record_hashes || null,
      executed: Boolean(phase4Result?.analysis)
    },
    audit: auditSummary,
    provenance: provenanceRecords,
    findings: anomalyFindings,
    checkpoints: {
      phase3: phase3Result?.hash_checkpoint,
      phase4: phase4Result?.hash_checkpoint || phase4?.hash_checkpoint,
      phase6: analysisResult?.phase6?.hash_checkpoint,
      phase7: analysisResult?.phase7?.hash_checkpoint,
      phase8: analysisResult?.phase8?.hash_checkpoint,
    },
    camera_frames: oodInputSource === "camera"
      ? cameraResults.map(frame => ({
          frame_index: frame.frame_index,
          source: frame.source,
          phase6: frame.phase6,
          phase7: frame.phase7,
          phase8: frame.phase8,
          phase9: frame.phase9
        }))
      : null
  };

  if (!mode) return (
    <Shell>
      <Panel>
        <div className="tc-eyebrow">TRUSTCV</div><h1 className="tc-title">What would you like to verify?</h1>
        <p className="tc-lede">Establish model identity, inspect model integrity, and continue to inference assurance only after the required gates pass.</p>
        <div className="tc-choice-grid">
          <button type="button" className="tc-choice-card" onClick={() => { setMode("model"); setStep(1); }}><Cpu size={26}/><div><b>Model Verification</b><span>Identity, integrity, and inference assurance</span></div><ArrowRight size={18}/></button>
          <button type="button" className="tc-choice-card" onClick={() => { setMode("dataset"); setStep(1); }}><Database size={26}/><div><b>Dataset Verification</b><span>Integrity and provenance of a dataset</span></div><ArrowRight size={18}/></button>
        </div>
      </Panel>
    </Shell>
  );

  if (mode === "dataset") return (
    <Shell rail={<Rail steps={DATASET_STEPS} current={Math.max(step, 1)} />}>
      {step === 1 && <Panel><Header number={1} eyebrow="DATASET · SELECT" title="Select a dataset" subtitle="Choose the dataset you want to verify."/><SourcePicker onUpload={f => {setInputName(f.name);setStep(2);setBusy(true);setError("Dataset verification is reserved for the next integration stage.");setBusy(false);}} onExisting={() => setError("Existing dataset selection will be wired with persistent MIRAD dataset registration later.")} existingLabel="Choose existing" existingIcon={<FolderOpen size={22}/>} />{error && <div className="tc-error">{error}</div>}</Panel>}
      {step >= 2 && <Panel><Header number={2} eyebrow="DATASET · REPORT" title="Dataset verification" subtitle={inputName}/><div className="tc-placeholder">Dataset verification is reserved for the next integration stage.</div><Footer reset={reset}/></Panel>}
    </Shell>
  );

  const current = Math.min(Math.max(step, 1), MODEL_STEPS.length);

  return (
    <Shell rail={<Rail steps={MODEL_STEPS} current={current} />}>
      {step === 1 && <Panel>
        <Header number={1} eyebrow="MODEL · SELECT" title="Upload your model" subtitle="Trusted CV first inspects the model. It will then tell you which clean reference dataset is compatible before you upload it."/>
        <div className="tc-upload-stack">
          <UploadField label="AI Model" sub=".pt / .pth / supported model file" accept=".pt,.pth,.onnx,.bin" value={modelFile} onFile={setModelFile}/>
        </div>
        <div className="tc-info-note"><Database size={16}/><span>Reference data is not downloaded or guessed automatically. Model compatibility is determined first.</span></div>
        <div className="tc-footer"><button type="button" className="tc-primary-btn" disabled={!modelFile || busy} onClick={runPhase3}><ShieldCheck size={16}/> Inspect Model <ArrowRight size={16}/></button></div>
        <button type="button" className="tc-existing-link" onClick={() => setError("Choose Upload for the Round-1 model. Existing-model registration will be wired after the MIRAD persistent store is added.")}><FolderOpen size={15}/> Choose existing model</button>
        {error && <div className="tc-error">{error}</div>}
      </Panel>}

      {step === 2 && <Panel>
        {busy || !phase3Result ? <><Header number={3} eyebrow="PHASE 3 · TRUST & IDENTITY" title="Trust & Identity" subtitle={inputName}/>{busy ? <Loading phase={3} modelName={inputName}/> : <><div className="tc-error">{error || "Phase 3 did not return a result."}</div><div className="tc-footer"><button type="button" className="tc-ghost-btn" onClick={() => setStep(1)}><ArrowLeft size={15}/> Back</button><button type="button" className="tc-primary-btn" onClick={runPhase3}><RotateCcw size={15}/> Retry Phase 3</button></div></>}
        </> : <>
          <Header number={3} eyebrow="PHASE 3 · TRUST & IDENTITY REPORT" title="Trust & Identity" subtitle="Identity and available trust/provenance evidence. This is not a model-safety verdict."/>
          <div className="tc-meta"><span>{inputName}</span><span>{phase3Result.model_type || "Model"}</span></div>
          <div className="tc-summary"><div><span>SHA-256</span><b>{phase3Result.sha256 || "—"}</b></div><div><span>Decision</span><b>{phase3Passed ? (phase3Review ? "REVIEW" : "PASS") : "STOP"}</b></div></div>
          <HashCheckpoint checkpoint={phase3Result.hash_checkpoint}/>
          <div className="tc-section-label">IDENTITY CHECKS · CLICK TO INSPECT</div>
          <div className="tc-checklist">{phase3Checks.length ? phase3Checks.map((c,i)=><CheckCard key={c.id || i} label={c.label || `Check ${i+1}`} state={checkState(c)} detail={c.detail} evidence={c.evidence || c.data || {}}/>) : <CheckCard label="MIRAD verification" state={phase3Passed ? "pass":"fail"} detail={phase3Result.detail || "Phase 3 response received."} evidence={phase3Result.mirad || {}}/>}</div>
          <Banner status={phase3Passed ? (phase3Review ? "warn" : "trusted") : "risk"} text={phase3Passed ? (phase3Review ? "Identity checks completed, but provenance or signature evidence is unavailable. Review the limitations, then proceed to Phase 4." : "Model identity checks completed. Proceed to Phase 4 to provide the compatible clean reference data required for integrity analysis.") : "The required Phase 3 identity checks did not pass. The workflow stops here."}/>
          <div className="tc-report-actions"><DownloadButton filename={`TrustCV_Phase3_${inputName.replace(/[^a-z0-9._-]/gi,"_")}.pdf`} payload={phase3Report} label="Download Phase 3 PDF" pdf/>{phase3Passed && <button type="button" className="tc-primary-btn" onClick={() => setStep(3)}>Continue to Phase 4 <ArrowRight size={16}/></button>}</div>
          <Footer reset={reset}/>
        </>}
      </Panel>}

      {step === 3 && <Panel>
        {busy || !phase4Result ? <><Header number={4} eyebrow="PHASE 4 · MODEL INTEGRITY" title="Model Integrity Analysis" subtitle="Provide clean reference data compatible with the uploaded model, then run the model-integrity and backdoor analysis." />
          {busy ? <Loading phase={4} modelName={inputName}/> : <>
            <div className="tc-phase-transition"><CheckCircle2 size={19} color="#3DDC84"/><div><b>Phase 3 completed</b><span>Model identity and trust checks are complete. Phase 4 now requires clean reference data.</span></div></div>
            {datasetRequirements && <div className="tc-dataset-guidance tc-phase4-dataset">
              <div className="tc-dataset-guidance-head"><Database size={18}/><div><b>Clean reference dataset required</b><span>{datasetRequirements.recommended_dataset}</span></div></div>
              <p>{datasetRequirements.why}</p>
              <div className="tc-guidance-grid">
                <div><span>Task</span><b>{datasetRequirements.expected?.task || "—"}</b></div>
                <div><span>Expected input</span><b>{datasetRequirements.expected?.height && datasetRequirements.expected?.width ? `${datasetRequirements.expected.height} × ${datasetRequirements.expected.width}` : "Model-specific"}</b></div>
                <div><span>Classes</span><b>{datasetRequirements.expected?.num_classes ?? "Model-specific"}</b></div>
              </div>
              <div className="tc-info-note" style={{marginTop: 10}}><Database size={15}/><span>Quick demo mode: 15 images build the calibration baseline and 7 separate images are evaluated. This keeps the 5-minute demo responsive; use a larger held-out split for full validation.</span></div>
              <DatasetUploadField value={referenceDataset} onDataset={handleReferenceDataset}/>
              {compatibilityBusy && <div className="tc-compatibility-box warn"><Loader2 size={16} className="tc-spin"/><span>Inspecting dataset format, images, annotations, and class compatibility…</span></div>}
              {!compatibilityBusy && compatibilityResult && (() => {
                const compatibility = compatibilityResult.compatibility || compatibilityResult;
                const compatible = compatibility.compatible ?? compatibilityResult.passed ?? false;
                const disposition = String(compatibility.disposition || compatibilityResult.disposition || "").toLowerCase();
                const isReview = compatible && disposition === "review";
                const detail = compatibility.errors?.[0] || compatibility.warnings?.[0] || compatibilityResult.error || "Reference dataset passed the compatibility checks.";
                return (
                  <div className={`tc-compatibility-box ${compatible ? (isReview ? "warn" : "pass") : "fail"}`}>
                    {compatible ? <CheckCircle2 size={17}/> : <XCircle size={17}/>}
                    <div><b>{compatible ? (isReview ? "COMPATIBLE — REVIEW" : "COMPATIBLE") : "INCOMPATIBLE"}</b><span>{detail}</span></div>
                  </div>
                );
              })()}
              {!compatibilityBusy && compatibilityResult?.dataset && <div className="tc-dataset-facts">
                <div><span>DATASET FORMAT</span><b>{String(compatibilityResult.dataset.format || "unknown").replaceAll("_", " ").toUpperCase()}</b></div>
                <div><span>LABELS / ANNOTATIONS</span><b>{compatibilityResult.dataset.label_status === "present" ? "PRESENT" : compatibilityResult.dataset.label_status === "absent" ? "NOT PRESENT" : "NOT APPLICABLE"}</b></div>
                <div><span>REFERENCE IMAGES</span><b>{compatibilityResult.dataset.num_images ?? "—"}</b></div>
                {compatibilityResult.dataset.annotation_validation?.coverage !== undefined && <div><span>LABEL COVERAGE (SAMPLED)</span><b>{Math.round(compatibilityResult.dataset.annotation_validation.coverage * 100)}%</b></div>}
              </div>}
            </div>}
            {!datasetRequirements && <div className="tc-info-note"><Database size={16}/><span>The backend did not return dataset compatibility guidance. A compatible clean reference dataset is still required before Phase 4 can run.</span></div>}
            {error && <div className="tc-error">{error}</div>}
            <div className="tc-footer">
              <button type="button" className="tc-ghost-btn" onClick={() => setStep(2)}><ArrowLeft size={15}/> Phase 3 Report</button>
              <button type="button" className="tc-primary-btn" disabled={!referenceDataset || !phase3Result || compatibilityBusy || !(compatibilityResult?.compatibility?.compatible ?? compatibilityResult?.compatible ?? compatibilityResult?.passed)} onClick={runPhase4}><ShieldCheck size={16}/> {referenceDataset ? "Start Phase 4" : "Upload Reference Data"} <ArrowRight size={16}/></button>
            </div>
          </>}
        </> : <>
          <Header number={4} eyebrow="PHASE 4 · MODEL INTEGRITY REPORT" title="Model Integrity" subtitle="Detailed evidence from the configured backdoor and model-integrity checks."/>
          <HashCheckpoint checkpoint={phase4Result.hash_checkpoint || phase4?.hash_checkpoint}/>
          <div className="tc-meta"><span>{inputName}</span><span>{phase4Result.model_type || "Model"}</span></div>
          <Banner
            status={phase4Quarantined ? "risk" : phase4Accepted ? "trusted" : "warn"}
            text={
              phase4Quarantined
                ? "A model-integrity signal was detected. The model is quarantined and cannot continue."
                : yoloDemoContinue
                  ? "YOLO-specific Phase 4 backdoor detection is a declared placeholder in this integration build. Demo continuation is enabled so the existing Phase 6–9 YOLO assurance path can be evaluated independently."
                  : phase4Accepted
                    ? "No sufficiently anomalous behavior was detected relative to the supplied calibration reference. ACCEPT is a behavioral assessment, not proof that the model is backdoor-free."
                    : "An anomalous behavioral pattern was detected relative to the calibration reference. REVIEW is required; an anomaly is not by itself proof of a backdoor."
            }
          />
          <div className="tc-section-label">INTEGRITY CHECKS · CLICK TO INSPECT</div>
          <div className="tc-checklist">{phase4Flags.length ? phase4Flags.map((f,i)=><CheckCard key={i} label={f.check || `Integrity check ${i+1}`} state={checkState(f)} detail={f.reason || f.raw_disposition} evidence={f}/>) : <CheckCard label="Phase 4 integrity analysis" state={phase4Quarantined ? "fail":phase4Accepted ? "pass":"warn"} detail={phase4?.detail || "Phase 4 result received."} evidence={phase4 || {}}/>}</div>
          <div className="tc-section-label">KEY DETECTION EVIDENCE</div>
          <div className="tc-evidence-panel">
            {phase4Demo && <div className="tc-evidence-row"><span>MODE</span><strong>Quick Demo · {phase4Demo.calibration_images} calibration + {phase4Demo.evaluation_images} evaluation</strong></div>}
            <div className="tc-evidence-row"><span>MEAN TRACE SCORE</span><strong>{phase4Metrics.mean_trace_score !== undefined ? Number(phase4Metrics.mean_trace_score).toFixed(4) : "—"}</strong></div>
            <div className="tc-evidence-row"><span>P90 TRACE SCORE</span><strong>{phase4Metrics.p90_trace_score !== undefined ? Number(phase4Metrics.p90_trace_score).toFixed(4) : "—"}</strong></div>
            <div className="tc-evidence-row"><span>CALIBRATION</span><strong>{phase4Calibration.used ? `${phase4Calibration.reference_images} reference images · ${Math.round(Number(phase4Calibration.threshold || 0) * 100)}th percentile cutoff` : "Not calibrated"}</strong></div>
            <div className="tc-evidence-row"><span>SUSPICIOUS FRACTION</span><strong>{Math.round(phase4Suspicious * 100)}%</strong></div>
            <div className="tc-evidence-row"><span>INTERPRETATION</span><strong>Anomaly detection relative to the supplied behavioral baseline</strong></div>
          </div>
          <div className="tc-report-actions">
            <DownloadButton filename={`TrustCV_Phase4_${inputName.replace(/[^a-z0-9._-]/gi,"_")}.pdf`} payload={phase4Report} label="Download Phase 4 PDF" pdf/>
            {canContinueToInference && (
              <button type="button" className="tc-primary-btn" onClick={() => setStep(4)}>
                {yoloDemoContinue ? "Continue to Phase 6–9 (Demo)" : "Continue to Inference"} <ArrowRight size={16}/>
              </button>
            )}
          </div>
          {phase4Quarantined && <div className="tc-stop-note"><XCircle size={17}/><span>Workflow stopped. A quarantined model cannot proceed to Phase 6–9.</span></div>}
          <Footer reset={reset}/>
        </>}
      </Panel>}

      {step === 4 && canContinueToInference && <Panel>
        <Header
          number={5}
          eyebrow="INFERENCE · INPUT"
          title={phase4Result.model_type === "yolo" ? "Select CV input" : "Continue to Phase 6–9"}
          subtitle={
            phase4Result.model_type === "yolo"
              ? "Choose a test image or use the live camera. Phase 6–9 assurance runs on the supplied input after the Phase 4 gate."
              : "Phase 6–9 are explicit Round-1 placeholders for SmallCNN."
          }
        />
        {phase4Result.model_type === "smallcnn" ? <><div className="tc-phase-transition"><CheckCircle2 size={19} color="#3DDC84"/><div><b>Phase 4 accepted</b><span>The model passed the integrity gate.</span></div></div><button type="button" className="tc-primary-btn" onClick={() => runAnalysis("demo")}>Run Phase 6–9 Demo <ArrowRight size={16}/></button></> : <div className="tc-input-options"><div className="tc-input-option"><div className="tc-input-option-head"><Upload size={18}/><div><b>Single image</b><span>Run one image through Phase 6–9.</span></div></div><SourcePicker onUpload={f => runAnalysis("image",f)} onExisting={() => setError("Use Upload for a YOLO test image in this integration build.")} existingLabel="Upload image" existingIcon={<FileStack size={22}/>} /></div><div className="tc-input-option"><div className="tc-input-option-head"><Camera size={18}/><div><b>Live camera</b><span>Capture until Stop, then analyze 6–9 frames.</span></div></div><CameraCapture onComplete={runCameraAnalysis}/></div></div>} {error && <div className="tc-error">{error}</div>}
      </Panel>}

      {step === 5 && busy && <Panel><Header number={6} eyebrow="INFERENCE · RUNNING" title="Running assurance phases" subtitle="Processing the supplied input."/><Loading phase={6} modelName={inputName}/></Panel>}

      {step === 6 && <Panel>
        <Header number={7} eyebrow="COMPLETE INFERENCE REPORT" title="TrustCV Verification Result" subtitle="A consolidated report of the completed assurance stages."/>
        <div className="tc-result-grid">
          <div><span>Phase 3 · Trust & Identity</span><StatusIcon state={phase3Passed ? "pass":"fail"}/></div>
          <div><span>Phase 4 · Model Integrity</span><StatusIcon state={phase4Quarantined ? "fail":phase4Accepted ? "pass":"warn"}/></div>
          <div><span>Phase 6 · OOD / Shift</span><StatusIcon state={phase6State(phase4Result?.model_type, oodResult, shiftResult)}/></div>
          <div><span>Phase 7 · Inference</span><StatusIcon state={phase7State(phase4Result?.model_type, phase4Result?.analysis?.phase7)}/></div>
          <div><span>Phase 8 · Integrity</span><StatusIcon state={phase8State(phase4Result?.analysis?.phase8)}/></div>
          <div><span>Phase 9 · Audit</span><StatusIcon state={phase4Result?.model_type === "smallcnn" ? "warn" : phase4Result?.analysis?.phase9?.status === "real" ? "pass" : "warn"}/></div>
        </div>
        {cameraResults.length > 0 && <div className="tc-camera-results-section">
          <div className="tc-section-label">LIVE CAMERA · FRAME RESULTS</div>
          <div className="tc-camera-results-grid">
            {cameraResults.map((frame, idx) => {
              const local = cameraFrames[idx], preview = local?.previewUrl || null;
              const dets = frame.phase7?.detections || [];
              const ood = frame.phase6?.inDistribution === false;
              return <div className="tc-camera-result-card" key={`${frame.frame_index}-${idx}`}>
                {preview && <img src={preview} alt={`Camera frame ${frame.frame_index}`} />}
                <div className="tc-camera-result-head"><b>Frame {frame.frame_index}</b><StatusIcon state={frame.error ? "fail" : ood ? "warn" : "pass"} size={16}/></div>
                <div className="tc-camera-result-meta"><span>{dets.length} detection{dets.length === 1 ? "" : "s"}</span><span>{ood ? "OOD / SHIFT" : "IN DISTRIBUTION"}</span></div>
                {dets.length > 0 && <div className="tc-camera-detections">{dets.map((d,j) => <div key={j}><span>{d.class_name}</span><b>{(Number(d.confidence)*100).toFixed(1)}%</b></div>)}</div>}
                {frame.error && <div className="tc-error">{frame.error}</div>}
              </div>;
            })}
          </div>
        </div>}
        <div className="tc-section-label">PHASE DETAILS · CLICK TO INSPECT</div>
        <div className="tc-phase-stack">
          <PhaseCard phase="Phase 3" title="Trust & Identity" state={phase3Passed ? "pass":"fail"} detail={`${phase3Checks.length} identity checks · click to inspect`}><div className="tc-card-inner"><PhaseInfo phase={3} result={phase3Result} modelType={phase4Result?.model_type}/><HashCheckpoint checkpoint={phase3Result?.hash_checkpoint}/>{phase3Checks.map((c,i)=><CheckCard key={c.id||i} label={c.label||`Check ${i+1}`} state={checkState(c)} detail={c.detail} evidence={c.evidence||c.data||{}}/>)}</div></PhaseCard>
          <PhaseCard phase="Phase 4" title="Model Integrity" state={phase4Quarantined ? "fail":phase4Accepted ? "pass":"warn"} detail={`TRACE behavioral analysis · ${phase4?.disposition || "unknown"}`}><div className="tc-card-inner"><PhaseInfo phase={4} result={phase4}/><HashCheckpoint checkpoint={phase4Result?.hash_checkpoint || phase4?.hash_checkpoint}/>{phase4Flags.map((f,i)=><CheckCard key={i} label={f.check||`Integrity check ${i+1}`} state={checkState(f)} detail={f.reason||f.raw_disposition} evidence={f}/>)}</div></PhaseCard>
          <PhaseCard phase="Phase 6" title="Distribution / OOD" state={phase6State(phase4Result?.model_type, oodResult, shiftResult)} detail={shiftResult?.detail || oodResult?.detail || "Distribution result not available"}><div className="tc-card-inner"><PhaseInfo phase={6} result={{...(oodResult||{}),shift:shiftResult}} modelType={phase4Result?.model_type} cameraCount={cameraResults.length}/><HashCheckpoint checkpoint={analysisResult?.phase6?.hash_checkpoint}/><CheckCard label="OOD / distribution result" state={phase6State(phase4Result?.model_type, oodResult, shiftResult)} detail={shiftResult?.detail || oodResult?.detail || "Distribution evidence not available"} evidence={oodResult||{}}/></div></PhaseCard>
          <PhaseCard phase="Phase 7" title="Inference" state={phase7State(phase4Result?.model_type, phase4Result?.analysis?.phase7)} detail={phase7Detail(phase4Result?.model_type, phase4Result?.analysis?.phase7)}><div className="tc-card-inner"><PhaseInfo phase={7} result={phase4Result?.analysis?.phase7} modelType={phase4Result?.model_type}/><HashCheckpoint checkpoint={analysisResult?.phase7?.hash_checkpoint}/><CheckCard label="Inference result" state={phase7State(phase4Result?.model_type, phase4Result?.analysis?.phase7)} detail={phase7Detail(phase4Result?.model_type, phase4Result?.analysis?.phase7)} evidence={phase4Result?.analysis?.phase7||{}}/></div></PhaseCard>
          <PhaseCard phase="Phase 8" title="Inference Integrity" state={phase8State(phase4Result?.analysis?.phase8)} detail={phase8Detail(phase4Result?.analysis?.phase8)}><div className="tc-card-inner"><PhaseInfo phase={8} result={phase4Result?.analysis?.phase8} modelType={phase4Result?.model_type}/><HashCheckpoint checkpoint={analysisResult?.phase8?.hash_checkpoint}/><CheckCard label="Integrity result" state={phase8State(phase4Result?.analysis?.phase8)} detail={phase8Detail(phase4Result?.analysis?.phase8)} evidence={phase4Result?.analysis?.phase8||{}}/></div></PhaseCard>
          <PhaseCard phase="Phase 9" title="Audit Ledger" state={phase4Result?.model_type === "smallcnn" ? "warn" : phase4Result?.analysis?.phase9?.ledger_verification === false ? "fail" : phase4Result?.analysis?.phase9?.status === "real" ? "pass" : "warn"} detail={phase4Result?.analysis?.phase9?.ledger_verification ? "MIRAD audit chain verified · click to inspect" : "Audit result not available"}><div className="tc-card-inner"><PhaseInfo phase={9} result={phase4Result?.analysis?.phase9} modelType={phase4Result?.model_type} cameraCount={cameraResults.length}/><CheckCard label="Audit chain verification" state={phase4Result?.analysis?.phase9?.ledger_verification === true ? "pass" : phase4Result?.analysis?.phase9?.ledger_verification === false ? "fail" : "warn"} detail={phase4Result?.analysis?.phase9?.ledger_message || "Audit evidence not available"} evidence={phase4Result?.analysis?.phase9||{}}/><div className="tc-report-actions"><RunPdfButton runId={phase4Result?.run_id} endpoint="/api/audit/pdf" filename={`TrustCV_Audit_Log_${phase4Result?.run_id || "run"}.pdf`} label="Download Audit Log PDF"/></div></div></PhaseCard>
        </div>
        <Banner
          status={
            phase4Quarantined
              ? "risk"
              : phase8State(phase4Result?.analysis?.phase8) === "fail"
                ? "risk"
                : phase6State(phase4Result?.model_type, oodResult, shiftResult) === "fail"
                  ? "risk"
                  : phase4Result?.model_type === "smallcnn"
                    ? "warn"
                    : "trusted"
          }
          text={
            phase4Quarantined
              ? "The model was quarantined during Phase 4. Downstream inference should not be interpreted as a clean assurance result."
              : phase8State(phase4Result?.analysis?.phase8) === "fail"
                ? "Phase 8 flagged the inference/output as requiring quarantine or review. The result is not fully trusted."
                : phase6State(phase4Result?.model_type, oodResult, shiftResult) === "fail"
                  ? "Phase 6 detected a distribution-shift or OOD condition for the supplied input."
                  : phase4Result?.model_type === "smallcnn"
                    ? "Phase 3 and real Phase 4 completed. Phase 6–9 remain explicit Round-1 placeholders."
                    : "The available inference assurance path completed; each phase above reflects its actual result."
          }
        />
        <div className="tc-report-actions">
          <DownloadButton filename={`TrustCV_Complete_Inference_${inputName.replace(/[^a-z0-9._-]/gi,"_")}.pdf`} payload={completeReport} label="Download Complete PDF" pdf/>
          <DownloadButton filename={`TrustCV_Complete_Inference_${inputName.replace(/[^a-z0-9._-]/gi,"_")}.json`} payload={completeReport} label="JSON Evidence"/>
          <button type="button" className="tc-primary-btn" onClick={() => setStep(7)}><ArrowRight size={15}/> Model Provenance</button>
        </div>
        <Footer reset={reset}/>
      </Panel>}

      {step === 7 && <Panel>
        <Header number={8} eyebrow="MODEL PROVENANCE" title="Model Provenance" subtitle="Lifecycle lineage from model identity through supplied inputs, inference outputs and security findings."/>
        <div className="tc-summary"><div><span>MODEL</span><b>{inputName || "—"}</b></div><div><span>RUN ID</span><b>{phase4Result?.run_id || "—"}</b></div></div>
        <div className="tc-section-label">MODEL PROVENANCE CARD</div>
        <div className="tc-provenance-card"><div><span>Model type</span><b>{phase4Result?.model_type || "—"}</b></div><div><span>SHA-256</span><b>{phase4Result?.sha256 || "—"}</b></div><div><span>MIRAD artifact</span><b>{phase3Result?.mirad?.artifact_identity?.artifact_id || phase3Result?.mirad?.evidence?.candidate?.artifact_id || `model:${String(inputName || "model").split(".")[0]}`}</b></div><div><span>Version</span><b>{phase3Result?.mirad?.artifact_identity?.version || phase3Result?.mirad?.evidence?.candidate?.version || "1.0.0"}</b></div><div><span>Audit chain</span><b>{auditSummary?.ledger_verification === true ? "VERIFIED" : "NOT VERIFIED"}</b></div><div><span>Digital signature</span><b>NOT AVAILABLE</b></div></div>
        <div className="tc-report-actions"><RunPdfButton runId={phase4Result?.run_id} endpoint="/api/provenance/pdf" filename={`TrustCV_Model_Provenance_${phase4Result?.run_id || "run"}.pdf`} label="Download Model Provenance PDF"/></div>
        <div className="tc-section-label">INPUT → OUTPUT TRACEABILITY</div>
        <div className="tc-provenance-list">{provenanceRecords.length ? provenanceRecords.map((p,i) => { const m=p.metadata||{}; const replay=p.replay_verification||{}; return <div className="tc-provenance-item" key={p.event_id || i}><b>{m.frame_index !== null && m.frame_index !== undefined ? `Frame ${m.frame_index}` : "Input"}</b><span>Input: {p.input_digest || "—"}</span><span>→ Model: {p.model_digest || "—"}</span><span>→ Phase 6: {m.phase6?.inDistribution === false ? "OOD / shift" : "In distribution / no OOD signal"}</span><span>→ Phase 7: {Array.isArray(m.phase7?.detections) ? `${m.phase7.detections.length} detection(s)` : "Recorded"}</span><span>→ Phase 8: {m.phase8?.integrity?.disposition || m.phase8?.disposition || "Recorded"}</span><span>→ Output: {p.output_digest || "—"}</span><span>→ Audit: {m.audit_event_id || "—"}</span><span>→ Replay: {replay.valid === true ? "VALID" : replay.valid === false ? "INVALID" : "NOT AVAILABLE"}</span></div>; }) : <div className="tc-placeholder">No provenance records are available yet.</div>}</div>
        <div className="tc-section-label">ANOMALOUS / MALICIOUS-OUTPUT CANDIDATES</div>
        {anomalyFindings.length ? <div className="tc-provenance-list">{anomalyFindings.map((f,i)=><div className="tc-provenance-item" key={f.finding_id||i}><b>{f.affected_asset || "Input"} · {f.recommended_disposition || "REVIEW"}</b><span>{f.reason}</span><span>Provenance: {f.provenance_id || "—"}</span><span>Audit: {f.audit_event_id || "—"}</span></div>)}</div> : <div className="tc-placeholder">No evidence-based anomaly candidates were recorded for this run.</div>}
        <div className="tc-section-label">COVERAGE & LIMITATIONS</div>
        <div className="tc-limitations"><div>✓ MIRAD artifact identity and SHA-256 verification</div><div>✓ Model hash re-verification after Phases 3, 4, 6, 7 and 8</div><div>✓ Hash-chained Phase 9 audit events with chain verification</div><div>✓ Input, model, output and audit-event lineage</div><div>• Hash continuity does not prove a model was benign before upload.</div><div>• Registration means reference-known identity, not model safety.</div><div>• Camera runs have no ground-truth labels; accuracy is not fabricated.</div><div>• OOD assesses input distribution; it does not prove that the expected object is present.</div><div>• TRACE is a declared TRACE-inspired behavioral adaptation with demo calibration limits.</div><div>• Digital signatures are intentionally deferred; no signer is fabricated.</div></div>
        <div className="tc-footer"><button type="button" className="tc-ghost-btn" onClick={() => setStep(6)}><ArrowLeft size={15}/> Complete Report</button><Footer reset={reset}/></div>
      </Panel>}

      {error && step !== 2 && step !== 3 && <div className="tc-error">{error}</div>}
    </Shell>
  );
}

function Shell({ children, rail }) {
  return (
    <div className="tc-root">
      <style>{CSS}</style>
      <div className="tc-header">
        <ShieldCheck size={20} color="#4FD1B3" />
        <span className="tc-header-title">TrustCV</span>
        <span className="tc-header-caption">MODEL ASSURANCE</span>
      </div>
      <div className="tc-body">
        {rail && <aside className="tc-rail-wrap">{rail}</aside>}
        <main className="tc-content">{children}</main>
      </div>
    </div>
  );
}

const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
.tc-root{--bg:#11151B;--panel:#171C24;--border:#28303B;--text:#EDEFF3;--muted:#8B93A3;--accent:#4FD1B3;--pass:#3DDC84;--fail:#FF6B6B;--warn:#FFB454;font-family:'Space Grotesk',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;width:100vw;padding:28px 34px;box-sizing:border-box}.tc-header{display:flex;align-items:center;gap:8px;width:100%;margin:0 0 28px;box-sizing:border-box}.tc-header-title{font-size:15px;font-weight:600}.tc-header-caption{font:10px 'IBM Plex Mono',monospace;color:var(--muted);letter-spacing:.08em;margin-left:5px}.tc-body{display:flex;flex-direction:row;align-items:stretch;gap:34px;width:100%;margin:0;box-sizing:border-box;min-height:calc(100vh - 88px)}.tc-rail-wrap{display:block;flex:0 0 150px;width:150px;min-width:150px;padding-top:8px;box-sizing:border-box}.tc-content{display:block;flex:1 1 auto;width:auto;min-width:0;max-width:none;box-sizing:border-box}.tc-panel{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:34px;box-shadow:0 12px 35px rgba(0,0,0,.12);width:100%;min-height:100%;box-sizing:border-box}.tc-phase-header{display:flex;gap:16px;align-items:flex-start;margin-bottom:24px}.tc-phase-number{font:12px 'IBM Plex Mono',monospace;color:var(--accent);border:1px solid rgba(79,209,179,.35);border-radius:6px;padding:6px 7px}.tc-eyebrow,.tc-section-label{font:10.5px 'IBM Plex Mono',monospace;color:var(--accent);letter-spacing:.06em}.tc-eyebrow{margin-bottom:7px}.tc-title{font-size:23px;font-weight:600;margin:0 0 10px;letter-spacing:-.02em}.tc-lede{font-size:13.5px;line-height:1.65;color:var(--muted);margin:0}.tc-choice-grid,.tc-upload-stack,.tc-checklist,.tc-phase-stack{display:flex;flex-direction:column;gap:9px}.tc-choice-card{display:flex;align-items:center;gap:18px;text-align:left;background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:20px 22px;min-height:82px;color:var(--text);cursor:pointer;box-sizing:border-box}.tc-choice-card:hover,.tc-upload-field:hover,.tc-source-card:hover{border-color:var(--accent)}.tc-choice-card b{display:block;font-size:15px}.tc-choice-card span{display:block;color:var(--muted);font-size:12px;margin-top:3px}.tc-choice-card>svg:last-child{margin-left:auto;color:var(--muted)}.tc-upload-stack{margin:18px 0}.tc-upload-field{display:flex;align-items:center;gap:12px;padding:14px;background:var(--bg);border:1px dashed var(--border);border-radius:9px;cursor:pointer}.tc-upload-icon{width:35px;height:35px;border-radius:7px;display:flex;align-items:center;justify-content:center;background:rgba(79,209,179,.08);color:var(--accent)}.tc-upload-copy{min-width:0;flex:1}.tc-upload-label{font-size:13.5px;font-weight:500}.tc-upload-sub{font-size:11.5px;color:var(--muted);margin-top:3px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.tc-upload-action{font:10.5px 'IBM Plex Mono',monospace;color:var(--accent)}.tc-hidden-input{display:none}.tc-source-grid{display:flex;gap:12px}.tc-source-card{flex:1;display:flex;flex-direction:column;align-items:flex-start;gap:6px;background:var(--bg);border:1px dashed var(--border);border-radius:9px;padding:21px;color:var(--text);cursor:pointer}.tc-source-card span:nth-of-type(1){font-size:14px;font-weight:500;margin-top:4px}.tc-source-sub{font-size:12px!important;color:var(--muted)!important}.tc-rail{display:flex;flex-direction:column}.tc-rail-item{display:flex;align-items:center;position:relative;min-height:48px}.tc-rail-dot{width:25px;height:25px;border-radius:50%;border:1px solid var(--border);display:flex;align-items:center;justify-content:center;font:10px 'IBM Plex Mono',monospace;color:var(--muted);background:var(--panel);z-index:1}.tc-rail-dot.active{border-color:var(--accent);color:var(--accent)}.tc-rail-dot.done{border-color:var(--pass);color:var(--pass);background:rgba(61,220,132,.08)}.tc-rail-label{font-size:11.5px;color:var(--muted);margin-left:10px;white-space:nowrap}.tc-rail-label.active{color:var(--text)}.tc-rail-line{position:absolute;left:12px;top:25px;width:1px;height:48px;background:var(--border)}.tc-rail-line.done{background:var(--pass)}.tc-meta{display:flex;justify-content:space-between;gap:12px;background:var(--bg);border:1px solid var(--border);border-radius:7px;padding:9px 12px;margin-bottom:15px;font:10.5px 'IBM Plex Mono',monospace;color:var(--muted)}.tc-summary{display:grid;grid-template-columns:2fr 1fr;gap:9px;margin-bottom:22px}.tc-summary>div{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:13px}.tc-summary span{display:block;font-size:9.5px;text-transform:uppercase;color:var(--muted);margin-bottom:6px}.tc-summary b{font:11.5px 'IBM Plex Mono',monospace;word-break:break-all}.tc-section-label{margin:21px 0 9px;color:var(--muted)}.tc-checkcard,.tc-phase-card{background:var(--bg);border:1px solid var(--border);border-radius:8px;overflow:hidden}.tc-checkcard.open,.tc-phase-card.open{border-color:#3A4553}.tc-checkcard-button,.tc-phase-card-button{width:100%;display:flex;align-items:center;gap:11px;background:transparent;border:0;color:var(--text);text-align:left;cursor:pointer;padding:14px}.tc-checkcard-button:disabled{cursor:default}.tc-checkcard-main{flex:1;min-width:0}.tc-checkcard-label{display:block;font-size:13px;font-weight:500}.tc-checkcard-summary,.tc-phase-card-detail{display:block;font:10.5px 'IBM Plex Mono',monospace;color:var(--muted);margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.tc-checkcard-detail,.tc-phase-card-body{border-top:1px solid var(--border);padding:13px 15px}.tc-evidence-note{font-size:12px;color:var(--muted);line-height:1.55;margin-bottom:8px}.tc-evidence-row{display:grid;grid-template-columns:150px 1fr;gap:12px;padding:6px 0;border-bottom:1px solid rgba(40,48,59,.7)}.tc-evidence-row:last-child{border:0}.tc-evidence-row span{font:10px 'IBM Plex Mono',monospace;color:var(--muted);text-transform:capitalize}.tc-evidence-row strong{font:10.5px 'IBM Plex Mono',monospace;font-weight:400;white-space:pre-wrap;word-break:break-word}.tc-phase-card-button{justify-content:space-between}.tc-phase-card-button>div:first-child{min-width:0}.tc-phase-card-phase{font:10px 'IBM Plex Mono',monospace;color:var(--accent);letter-spacing:.05em}.tc-phase-card-title{font-size:13.5px;font-weight:500;margin-top:4px}.tc-phase-card-right{display:flex;align-items:center;gap:9px;flex-shrink:0}.tc-card-inner{padding-top:2px}.tc-banner{border:1px solid;border-radius:8px;padding:15px 17px;margin:18px 0}.tc-banner-top{display:flex;align-items:center;gap:8px;font:11px 'IBM Plex Mono',monospace;letter-spacing:.05em}.tc-banner p{font-size:12.5px;line-height:1.55;margin:7px 0 0;color:var(--text);opacity:.86}.tc-report-actions{display:flex;justify-content:flex-end;align-items:center;gap:9px;flex-wrap:wrap;margin-top:18px}.tc-primary-btn,.tc-ghost-btn,.tc-secondary-btn{display:flex;align-items:center;gap:8px;font:500 13px 'Space Grotesk',sans-serif;border-radius:7px;padding:10px 15px;cursor:pointer}.tc-primary-btn{background:var(--accent);border:1px solid var(--accent);color:#0B1310}.tc-primary-btn:disabled{opacity:.4;cursor:not-allowed}.tc-ghost-btn{background:transparent;border:1px solid var(--border);color:var(--muted)}.tc-ghost-btn:hover,.tc-secondary-btn:hover{border-color:var(--accent);color:var(--text)}.tc-secondary-btn:disabled{opacity:.55;cursor:wait}.tc-secondary-btn{background:transparent;border:1px solid rgba(79,209,179,.35);color:var(--accent)}.tc-footer{display:flex;justify-content:flex-end;gap:9px;margin-top:20px}.tc-phase-transition{display:flex;align-items:flex-start;gap:11px;padding:14px;background:rgba(61,220,132,.05);border:1px solid rgba(61,220,132,.2);border-radius:8px}.tc-phase-transition b{display:block;font-size:13px}.tc-phase-transition span{display:block;color:var(--muted);font-size:11.5px;margin-top:3px}.tc-stop-note{display:flex;gap:9px;align-items:center;color:var(--fail);font-size:12px;border:1px solid rgba(255,107,107,.2);background:rgba(255,107,107,.04);border-radius:7px;padding:12px;margin-top:12px}.tc-loading{text-align:center;padding:34px 18px 22px}.tc-loader{color:var(--accent);margin-bottom:15px}.tc-loading-title{font-size:17px;font-weight:600}.tc-loading-model{font:10.5px 'IBM Plex Mono',monospace;color:var(--muted);margin-top:5px;word-break:break-all}.tc-progress{height:3px;background:var(--border);border-radius:99px;overflow:hidden;max-width:430px;margin:22px auto 15px}.tc-progress div{height:100%;width:35%;background:var(--accent);animation:tc-slide 1.3s ease-in-out infinite}.tc-loading-stages{display:flex;justify-content:center;flex-wrap:wrap;gap:7px}.tc-loading-stages span{font:9.5px 'IBM Plex Mono',monospace;color:var(--muted);border:1px solid var(--border);border-radius:5px;padding:5px 7px}.tc-loading p{font-size:11.5px;color:var(--muted);line-height:1.5;max-width:55ch;margin:12px auto 0}.tc-result-grid{display:flex;flex-direction:column;margin-bottom:18px}.tc-result-grid>div{display:flex;justify-content:space-between;align-items:center;padding:12px 0;border-bottom:1px solid var(--border);font-size:13px}.tc-result-grid>div:last-child{border-bottom:0}.tc-placeholder{padding:17px;background:var(--bg);border:1px solid var(--border);border-radius:8px;color:var(--muted);font-size:13px}.tc-error{color:var(--fail);font:11.5px 'IBM Plex Mono',monospace;line-height:1.5;margin-top:12px}@keyframes tc-spin{to{transform:rotate(360deg)}}.tc-spin{animation:tc-spin 1s linear infinite}@keyframes tc-slide{0%{transform:translateX(-150%)}50%{transform:translateX(190%)}100%{transform:translateX(420%)}}
.tc-info-note{display:flex;align-items:flex-start;gap:9px;padding:12px 14px;background:rgba(79,209,179,.04);border:1px solid rgba(79,209,179,.18);border-radius:8px;color:var(--muted);font-size:11.5px;line-height:1.5}.tc-info-note svg{color:var(--accent);flex-shrink:0;margin-top:1px}.tc-phase4-dataset{margin-top:18px}.tc-dataset-guidance{margin-top:20px;padding:16px;background:rgba(79,209,179,.035);border:1px solid rgba(79,209,179,.18);border-radius:10px}.tc-dataset-guidance-head{display:flex;gap:10px;align-items:flex-start}.tc-dataset-guidance-head svg{color:var(--accent);margin-top:1px;flex-shrink:0}.tc-dataset-guidance-head b{display:block;font-size:13px}.tc-dataset-guidance-head span{display:block;color:var(--accent);font:10.5px 'IBM Plex Mono',monospace;margin-top:4px;line-height:1.5}.tc-dataset-guidance p{font-size:11.5px;line-height:1.55;color:var(--muted);margin:10px 0}.tc-guidance-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:11px 0}.tc-guidance-grid>div{background:var(--bg);border:1px solid var(--border);border-radius:7px;padding:9px}.tc-guidance-grid span{display:block;font-size:9px;color:var(--muted);text-transform:uppercase;margin-bottom:4px}.tc-guidance-grid b{display:block;font:10.5px 'IBM Plex Mono',monospace;word-break:break-word}.tc-dataset-selected{font:10.5px 'IBM Plex Mono',monospace;color:var(--accent);margin-top:8px}.tc-dataset-picker{margin-top:14px}.tc-dataset-picker-title{display:flex;gap:10px;align-items:flex-start;margin-bottom:10px}.tc-dataset-picker-title svg{color:var(--accent);margin-top:1px;flex-shrink:0}.tc-dataset-picker-title b{display:block;font-size:12.5px}.tc-dataset-picker-title span{display:block;color:var(--muted);font-size:10.5px;margin-top:3px;line-height:1.45}.tc-dataset-picker-actions{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}.tc-dataset-picker-actions .tc-source-card{min-height:105px}.tc-dataset-picker-actions button:disabled{opacity:.55;cursor:wait}.tc-dataset-picker .tc-dataset-selected{margin-top:8px}.tc-dataset-picker .tc-compatibility-box{margin-top:10px}.tc-dataset-picker + .tc-dataset-selected{display:none}@media(max-width:760px){.tc-root{padding:18px;width:100%;min-height:100vh}.tc-body{display:block}.tc-rail-wrap{margin-bottom:18px}.tc-rail{flex-direction:row;overflow-x:auto;gap:5px}.tc-rail-item{min-height:auto}.tc-rail-line,.tc-rail-label{display:none}.tc-panel{padding:21px}.tc-title{font-size:20px}.tc-summary{grid-template-columns:1fr}.tc-source-grid{flex-direction:column}.tc-meta{flex-direction:column}.tc-evidence-row{grid-template-columns:1fr;gap:3px}.tc-report-actions{justify-content:stretch}.tc-report-actions>*{flex:1;justify-content:center}}
.tc-dataset-facts{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px;margin-top:10px}.tc-dataset-facts>div{border:1px solid #252c38;border-radius:9px;padding:10px 11px;background:#10141a}.tc-dataset-facts span{display:block;font-size:9px;letter-spacing:.08em;color:#7f8797}.tc-dataset-facts b{display:block;margin-top:5px;font-size:11px;color:#e5e8ee;word-break:break-word}.tc-compatibility-box{display:flex;gap:11px;align-items:flex-start;margin-top:14px;padding:13px 15px;border:1px solid;border-radius:10px}.tc-compatibility-box.pass{border-color:#3DDC84;background:rgba(61,220,132,.06);color:#3DDC84}.tc-compatibility-box.warn{border-color:#FFB454;background:rgba(255,180,84,.06);color:#FFB454}.tc-compatibility-box.fail{border-color:#FF6B6B;background:rgba(255,107,107,.06);color:#FF6B6B}.tc-compatibility-box b{display:block;font-size:12px;letter-spacing:.04em}.tc-compatibility-box span{display:block;color:#9ca4b4;font-size:12px;margin-top:4px;line-height:1.45}
.tc-input-options{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:18px}.tc-input-option{background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:15px}.tc-input-option-head{display:flex;gap:9px;align-items:flex-start;margin-bottom:12px}.tc-input-option-head svg{color:var(--accent);margin-top:1px;flex-shrink:0}.tc-input-option-head b{display:block;font-size:13px}.tc-input-option-head span{display:block;color:var(--muted);font-size:11px;margin-top:3px;line-height:1.4}.tc-camera-box{margin-top:4px}.tc-camera-preview{position:relative;aspect-ratio:16/9;background:#0B0F14;border:1px solid var(--border);border-radius:8px;overflow:hidden}.tc-camera-video{width:100%;height:100%;object-fit:cover;display:block}.tc-camera-placeholder{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;color:var(--muted);gap:7px}.tc-camera-placeholder svg{color:var(--accent)}.tc-camera-placeholder span{font-size:12px}.tc-camera-placeholder small{font-size:10px}.tc-hidden-canvas{display:none}.tc-camera-controls{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-top:10px}.tc-camera-count{font:10px 'IBM Plex Mono',monospace;text-align:right}.tc-camera-count span{display:block;color:var(--muted);font-size:8.5px}.tc-camera-count b{display:block;color:var(--accent);margin-top:3px}.tc-camera-help{font-size:10.5px;color:var(--muted);line-height:1.45;margin-top:9px}.tc-camera-results-section{margin-top:20px}.tc-camera-results-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.tc-camera-result-card{background:var(--bg);border:1px solid var(--border);border-radius:9px;overflow:hidden;padding-bottom:10px}.tc-camera-result-card img{display:block;width:100%;aspect-ratio:16/10;object-fit:cover;background:#0B0F14}.tc-camera-result-head{display:flex;justify-content:space-between;align-items:center;padding:10px 11px 4px;font-size:12px}.tc-camera-result-meta{display:flex;justify-content:space-between;gap:6px;padding:0 11px;color:var(--muted);font:9px 'IBM Plex Mono',monospace}.tc-camera-detections{padding:7px 11px 0}.tc-camera-detections>div{display:flex;justify-content:space-between;border-top:1px solid var(--border);padding:5px 0;font-size:10px}.tc-camera-detections b{font-family:'IBM Plex Mono',monospace;font-weight:400}.tc-camera-result-card .tc-error{padding:0 11px;font-size:9px}.tc-hash-checkpoint{display:flex;align-items:center;gap:10px;border:1px solid;border-radius:9px;padding:11px 13px;margin:10px 0 13px}.tc-hash-checkpoint.pass{border-color:rgba(61,220,132,.28);background:rgba(61,220,132,.05)}.tc-hash-checkpoint.fail{border-color:rgba(255,107,107,.35);background:rgba(255,107,107,.05)}.tc-hash-checkpoint-main{flex:1;min-width:0}.tc-hash-checkpoint-main b{display:block;font:10.5px 'IBM Plex Mono',monospace}.tc-hash-checkpoint-main span{display:block;font-size:10.5px;color:var(--muted);margin-top:3px}.tc-hash-checkpoint-digest{font:9px 'IBM Plex Mono',monospace;color:var(--muted);max-width:150px;overflow:hidden;text-overflow:ellipsis}.tc-provenance-card{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px}.tc-provenance-card>div{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:12px}.tc-provenance-card span{display:block;font-size:9px;color:var(--muted);text-transform:uppercase;margin-bottom:5px}.tc-provenance-card b{font:10.5px 'IBM Plex Mono',monospace;word-break:break-all}.tc-provenance-list{display:flex;flex-direction:column;gap:8px}.tc-provenance-item{display:flex;flex-direction:column;gap:5px;background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:12px}.tc-provenance-item b{font-size:12px}.tc-provenance-item span{font:9.5px 'IBM Plex Mono',monospace;color:var(--muted);word-break:break-all}.tc-limitations{display:flex;flex-direction:column;gap:7px;background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:13px;font-size:11.5px;color:var(--muted);line-height:1.45}.tc-limitations div:first-child,.tc-limitations div:nth-child(2),.tc-limitations div:nth-child(3),.tc-limitations div:nth-child(4){color:var(--text)}
.tc-phase-metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin:2px 0 12px}.tc-phase-metric{background:#10141a;border:1px solid #252c38;border-radius:7px;padding:9px 10px;min-width:0}.tc-phase-metric span{display:block;font:8.5px 'IBM Plex Mono',monospace;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin-bottom:5px}.tc-phase-metric b{display:block;font:10.5px 'IBM Plex Mono',monospace;color:var(--text);font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.tc-phase-metrics+.tc-checkcard{margin-top:4px}
@media(max-width:760px){.tc-phase-metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.tc-input-options{grid-template-columns:1fr}.tc-camera-results-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
`;

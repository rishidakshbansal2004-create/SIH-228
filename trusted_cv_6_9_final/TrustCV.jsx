import { useState, useRef, useCallback } from "react";
import {
  ShieldCheck, Database, Cpu, Upload, FolderOpen, Camera, FileStack,
  CheckCircle2, XCircle, AlertTriangle, Loader2, ArrowRight, ArrowLeft, RotateCcw
} from "lucide-react";

/* ============================================================================
   INTEGRATION LAYER
   ----------------------------------------------------------------------------
   This is the ONLY place that should change when you wire in real code.
   Every function below currently returns mocked, delayed results shaped
   exactly like what the UI expects. Replace the body of each function with
   a real call (fetch to your backend, a local model run, a Python bridge,
   whatever) but keep the return shape the same and the UI needs no changes.

   Each function receives the raw input the user provided (a File object or
   a string id of an existing item) and must return a Promise resolving to:

     verifyDataset(input) / verifyModel(input) ->
       {
         passed: boolean,
         checks: [{ id, label, passed, detail? }, ...]
       }

     runOODDetection(model, input) ->
       { inDistribution: boolean, confidence: number /* 0-1 */, detail?: string }

     runDistributionShift(model, input) ->
       { shiftDetected: boolean, severity: "none"|"low"|"medium"|"high", detail?: string }
   ========================================================================== */

const DELAY = (ms) => new Promise((res) => setTimeout(res, ms));

const VerificationAPI = {
  async verifyDataset(_input) {
    // TODO: replace with real dataset verification call, e.g.:
    // const res = await fetch("/api/verify/dataset", { method: "POST", body: form });
    // return await res.json();
    await DELAY(1600);
    const checks = [
      { id: "hash", label: "Hash Verification", passed: true, detail: "sha256:9f2a...c31e matches manifest" },
      { id: "sig", label: "Digital Signature", passed: true, detail: "Signed by known publisher key" },
      { id: "provenance", label: "Dataset Provenance", passed: true, detail: "Chain of custody intact" },
      { id: "integrity", label: "Integrity / Tampering Check", passed: true, detail: "No modified records detected" },
    ];
    return { passed: checks.every((c) => c.passed), checks };
  },

  async verifyModel(_input) {
    // TODO: replace with real model verification call.
    await DELAY(1800);
    const checks = [
      { id: "vendor", label: "Vendor Authentication", passed: true, detail: "Vendor cert valid" },
      { id: "hash", label: "Model Hash Verification", passed: true, detail: "sha256:71bd...0a44 matches release" },
      { id: "sig", label: "Digital Signature", passed: true, detail: "Signature valid" },
      { id: "version", label: "Version Verification", passed: true, detail: "v2.3.1, latest stable" },
      { id: "provenance", label: "Model Provenance", passed: true, detail: "Training lineage confirmed" },
      { id: "tamper", label: "Tampering Detection", passed: true, detail: "Weights unmodified since release" },
    ];
    return { passed: checks.every((c) => c.passed), checks };
  },

  async runOODDetection(_model, _input) {
    // TODO: replace with a real OOD detector call.
    await DELAY(1400);
    return { inDistribution: true, confidence: 0.94, detail: "Input matches training domain" };
  },

  async runDistributionShift(_model, _input) {
    // TODO: replace with a real distribution-shift detector call.
    await DELAY(1400);
    return { shiftDetected: false, severity: "none", detail: "No significant divergence from training distribution" };
  },
};

/* ========================================================================== */

const STEPS_DATASET = ["Select", "Verify", "Result"];
const STEPS_MODEL = ["Select", "Verify", "Input", "Analyze", "Result"];

function StatusIcon({ state, size = 18 }) {
  if (state === "pass") return <CheckCircle2 size={size} color="#3DDC84" />;
  if (state === "fail") return <XCircle size={size} color="#FF6B6B" />;
  if (state === "warn") return <AlertTriangle size={size} color="#FFB454" />;
  if (state === "pending") return <Loader2 size={size} color="#8B93A3" className="tc-spin" />;
  return null;
}

function CheckRow({ label, state, detail }) {
  return (
    <div className="tc-checkrow">
      <div className="tc-checkrow-top">
        <span className="tc-checkrow-label">{label}</span>
        <StatusIcon state={state} />
      </div>
      {detail && <div className="tc-checkrow-detail">{detail}</div>}
    </div>
  );
}

function Rail({ steps, current }) {
  return (
    <div className="tc-rail">
      {steps.map((s, i) => {
        const idx = i + 1;
        const done = idx < current;
        const active = idx === current;
        return (
          <div className="tc-rail-item" key={s}>
            <div className={`tc-rail-dot ${done ? "done" : ""} ${active ? "active" : ""}`}>
              {done ? <CheckCircle2 size={14} /> : idx}
            </div>
            <span className={`tc-rail-label ${active ? "active" : ""}`}>{s}</span>
            {i < steps.length - 1 && <div className={`tc-rail-line ${done ? "done" : ""}`} />}
          </div>
        );
      })}
    </div>
  );
}

function SourcePicker({ onUpload, onExisting, existingLabel, existingIcon }) {
  const fileRef = useRef(null);
  return (
    <div className="tc-source-grid">
      <button className="tc-source-card" onClick={() => fileRef.current?.click()}>
        <Upload size={22} />
        <span>Upload</span>
        <span className="tc-source-sub">from your device</span>
      </button>
      <input
        ref={fileRef}
        type="file"
        className="tc-hidden-input"
        onChange={(e) => {
          if (e.target.files?.[0]) onUpload(e.target.files[0]);
        }}
      />
      <button className="tc-source-card" onClick={onExisting}>
        {existingIcon}
        <span>{existingLabel}</span>
        <span className="tc-source-sub">already registered</span>
      </button>
    </div>
  );
}

function Panel({ eyebrow, title, children }) {
  return (
    <div className="tc-panel">
      {eyebrow && <div className="tc-eyebrow">{eyebrow}</div>}
      {title && <h2 className="tc-title">{title}</h2>}
      {children}
    </div>
  );
}

function OverallBanner({ status, text }) {
  const map = {
    trusted: { bg: "rgba(61,220,132,0.1)", border: "#3DDC84", icon: <ShieldCheck size={20} color="#3DDC84" />, label: "TRUSTED" },
    risk: { bg: "rgba(255,107,107,0.1)", border: "#FF6B6B", icon: <XCircle size={20} color="#FF6B6B" />, label: "RISK DETECTED" },
    warn: { bg: "rgba(255,180,84,0.1)", border: "#FFB454", icon: <AlertTriangle size={20} color="#FFB454" />, label: "REVIEW RECOMMENDED" },
  };
  const m = map[status];
  return (
    <div className="tc-banner" style={{ background: m.bg, borderColor: m.border }}>
      <div className="tc-banner-top">
        {m.icon}
        <span className="tc-banner-label" style={{ color: m.border }}>{m.label}</span>
      </div>
      <p className="tc-banner-text">{text}</p>
    </div>
  );
}

export default function TrustCV() {
  const [mode, setMode] = useState(null); // "dataset" | "model"
  const [step, setStep] = useState(0);
  const [busy, setBusy] = useState(false);
  const [inputName, setInputName] = useState("");
  const [datasetResult, setDatasetResult] = useState(null);
  const [modelResult, setModelResult] = useState(null);
  const [oodInputSource, setOodInputSource] = useState(null); // "camera" | "dataset"
  const [oodResult, setOodResult] = useState(null);
  const [shiftResult, setShiftResult] = useState(null);
  const [error, setError] = useState(null);

  const reset = useCallback(() => {
    setMode(null); setStep(0); setBusy(false); setInputName("");
    setDatasetResult(null); setModelResult(null); setOodInputSource(null);
    setOodResult(null); setShiftResult(null); setError(null);
  }, []);

  const chooseMode = (m) => { setMode(m); setStep(1); };

  const runDatasetVerification = async (name) => {
    setInputName(name); setStep(2); setBusy(true); setError(null);
    try {
      const res = await VerificationAPI.verifyDataset(name);
      setDatasetResult(res);
    } catch (e) {
      setError("Dataset verification could not complete. Try again.");
    } finally {
      setBusy(false); setStep(3);
    }
  };

  const runModelVerification = async (name) => {
    setInputName(name); setStep(2); setBusy(true); setError(null);
    try {
      const res = await VerificationAPI.verifyModel(name);
      setModelResult(res);
    } catch (e) {
      setError("Model verification could not complete. Try again.");
    } finally {
      setBusy(false);
    }
  };

  const runOODAndShift = async (source) => {
    setOodInputSource(source); setStep(4); setBusy(true); setError(null);
    try {
      const [ood, shift] = await Promise.all([
        VerificationAPI.runOODDetection(inputName, source),
        VerificationAPI.runDistributionShift(inputName, source),
      ]);
      setOodResult(ood); setShiftResult(shift);
    } catch (e) {
      setError("Testing could not complete. Try again.");
    } finally {
      setBusy(false); setStep(5);
    }
  };

  // ---- render: selection ----
  if (!mode) {
    return (
      <Shell>
        <Panel eyebrow="TrustCV" title="What would you like to verify?">
          <p className="tc-lede">
            Choose a model to confirm its authenticity and test it against new data,
            or check a dataset's integrity and provenance on its own.
          </p>
          <div className="tc-choice-grid">
            <button className="tc-choice-card" onClick={() => chooseMode("model")}>
              <Cpu size={26} />
              <div>
                <div className="tc-choice-title">Model Verification</div>
                <div className="tc-choice-sub">Authenticity, integrity, and provenance of an AI model</div>
              </div>
              <ArrowRight size={18} className="tc-choice-arrow" />
            </button>
            <button className="tc-choice-card" onClick={() => chooseMode("dataset")}>
              <Database size={26} />
              <div>
                <div className="tc-choice-title">Dataset Verification</div>
                <div className="tc-choice-sub">Integrity, authenticity, and provenance of a dataset</div>
              </div>
              <ArrowRight size={18} className="tc-choice-arrow" />
            </button>
          </div>
        </Panel>
      </Shell>
    );
  }

  // ---- dataset flow ----
  if (mode === "dataset") {
    return (
      <Shell rail={<Rail steps={STEPS_DATASET} current={step} />}>
        {step === 1 && (
          <Panel eyebrow="Dataset · Step 1" title="Select a dataset">
            <SourcePicker
              onUpload={(f) => runDatasetVerification(f.name)}
              onExisting={() => runDatasetVerification("clinical-notes-v4 (registered)")}
              existingLabel="Choose existing"
              existingIcon={<FolderOpen size={22} />}
            />
          </Panel>
        )}
        {step >= 2 && (
          <Panel eyebrow="Dataset · Step 2" title="Verifying dataset">
            <div className="tc-target">{inputName}</div>
            <div className="tc-checklist">
              {["Hash Verification", "Digital Signature", "Dataset Provenance", "Integrity / Tampering Check"].map((label, i) => {
                const res = datasetResult?.checks?.[i];
                const state = !datasetResult ? "pending" : res.passed ? "pass" : "fail";
                return <CheckRow key={label} label={label} state={state} detail={res?.detail} />;
              })}
            </div>
            {datasetResult && (
              <>
                <OverallBanner
                  status={datasetResult.passed ? "trusted" : "risk"}
                  text={
                    datasetResult.passed
                      ? "All checks passed. This dataset's integrity and provenance are confirmed."
                      : "One or more checks failed. This dataset should not be trusted until the failing check is resolved."
                  }
                />
                <div className="tc-final-status" style={{ color: datasetResult.passed ? "#3DDC84" : "#FF6B6B" }}>
                  {datasetResult.passed ? "Dataset Verified — PASS" : "Dataset Verification Failed — FAIL"}
                </div>
              </>
            )}
            <FooterActions onReset={reset} showReset={!!datasetResult} />
          </Panel>
        )}
      </Shell>
    );
  }

  // ---- model flow ----
  let railCurrent = step;
  if (step >= 2 && !modelResult) railCurrent = 2;
  else if (modelResult && !modelResult.passed) railCurrent = 2;
  else if (step === 5) railCurrent = 5;

  return (
    <Shell rail={<Rail steps={STEPS_MODEL} current={railCurrent} />}>
      {step === 1 && (
        <Panel eyebrow="Model · Step 1" title="Select a model">
          <SourcePicker
            onUpload={(f) => runModelVerification(f.name)}
            onExisting={() => runModelVerification("rice-disease-classifier-v2 (registered)")}
            existingLabel="Choose existing"
            existingIcon={<FolderOpen size={22} />}
          />
        </Panel>
      )}

      {step >= 2 && !modelResult && (
        <Panel eyebrow="Model · Step 2" title="Verifying model">
          <div className="tc-target">{inputName}</div>
          <div className="tc-checklist">
            {["Vendor Authentication", "Model Hash Verification", "Digital Signature", "Version Verification", "Model Provenance", "Tampering Detection"].map((label) => (
              <CheckRow key={label} label={label} state="pending" />
            ))}
          </div>
        </Panel>
      )}

      {modelResult && !modelResult.passed && (
        <Panel eyebrow="Model · Step 2" title="Verification failed">
          <div className="tc-target">{inputName}</div>
          <div className="tc-checklist">
            {modelResult.checks.map((c) => (
              <CheckRow key={c.id} label={c.label} state={c.passed ? "pass" : "fail"} detail={c.detail} />
            ))}
          </div>
          <OverallBanner
            status="risk"
            text="Model Verification Failed. The workflow has stopped — resolve the failing check above before testing this model."
          />
          <FooterActions onReset={reset} showReset />
        </Panel>
      )}

      {modelResult && modelResult.passed && step === 2 && (
        <Panel eyebrow="Model · Step 2" title="Model verified">
          <div className="tc-target">{inputName}</div>
          <div className="tc-checklist">
            {modelResult.checks.map((c) => (
              <CheckRow key={c.id} label={c.label} state="pass" detail={c.detail} />
            ))}
          </div>
          <OverallBanner status="trusted" text="Model Verified Successfully. Continue to out-of-distribution and shift testing." />
          <div className="tc-footer">
            <button className="tc-primary-btn" onClick={() => setStep(3)}>
              Continue <ArrowRight size={16} />
            </button>
          </div>
        </Panel>
      )}

      {modelResult && modelResult.passed && step === 3 && (
        <Panel eyebrow="Model · Step 3" title="Select input source">
          <p className="tc-lede">Choose the data the verified model will be tested against.</p>
          <div className="tc-source-grid">
            <button className="tc-source-card" onClick={() => runOODAndShift("camera")}>
              <Camera size={22} />
              <span>Live Camera</span>
              <span className="tc-source-sub">test against a live feed</span>
            </button>
            <button className="tc-source-card" onClick={() => runOODAndShift("dataset")}>
              <FileStack size={22} />
              <span>Upload Dataset</span>
              <span className="tc-source-sub">test against a file</span>
            </button>
          </div>
        </Panel>
      )}

      {step >= 4 && (!oodResult || !shiftResult) && (
        <Panel eyebrow="Model · Step 4" title="Analyzing input">
          <div className="tc-target">Source: {oodInputSource === "camera" ? "Live Camera" : "Uploaded dataset"}</div>
          <div className="tc-checklist">
            <CheckRow label="OOD Detection" state="pending" />
            <CheckRow label="Distribution Shift Detection" state="pending" />
          </div>
        </Panel>
      )}

      {step === 5 && oodResult && shiftResult && (
        <Panel eyebrow="Model · Step 5" title="TrustCV Verification Result">
          <div className="tc-result-grid">
            <div className="tc-result-row">
              <span>Model Verification</span>
              <StatusIcon state="pass" />
            </div>
            <div className="tc-result-row">
              <span>OOD Detection</span>
              <StatusIcon state={oodResult.inDistribution ? "pass" : "fail"} />
            </div>
            <div className="tc-result-row">
              <span>Distribution Shift</span>
              <StatusIcon state={shiftResult.shiftDetected ? "warn" : "pass"} />
            </div>
          </div>

          <div className="tc-detail-block">
            <div className="tc-detail-line">
              {oodResult.inDistribution ? "In-Distribution" : "Out-of-Distribution"}
              {typeof oodResult.confidence === "number" && (
                <span className="tc-mono-score"> · confidence {(oodResult.confidence * 100).toFixed(1)}%</span>
              )}
            </div>
            <div className="tc-detail-line">
              {shiftResult.shiftDetected ? `Distribution Shift Detected · severity: ${shiftResult.severity}` : "No Significant Distribution Shift"}
            </div>
          </div>

          {(() => {
            const trusted = oodResult.inDistribution && !shiftResult.shiftDetected;
            return (
              <OverallBanner
                status={trusted ? "trusted" : "risk"}
                text={
                  trusted
                    ? "The model is authentic, verified, and the test input matches its expected distribution."
                    : "The model is authentic and verified, but the provided input differs from the expected data distribution. Results may therefore be unreliable."
                }
              />
            );
          })()}

          <FooterActions onReset={reset} showReset />
        </Panel>
      )}

      {error && <div className="tc-error">{error}</div>}
    </Shell>
  );
}

function FooterActions({ onReset, showReset }) {
  if (!showReset) return null;
  return (
    <div className="tc-footer">
      <button className="tc-ghost-btn" onClick={onReset}>
        <RotateCcw size={15} /> Run another verification
      </button>
    </div>
  );
}

function Shell({ children, rail }) {
  return (
    <div className="tc-root">
      <style>{CSS}</style>
      <div className="tc-header">
        <ShieldCheck size={20} color="#4FD1B3" />
        <span className="tc-header-title">TrustCV</span>
      </div>
      <div className="tc-body">
        {rail && <div className="tc-rail-wrap">{rail}</div>}
        <div className="tc-content">{children}</div>
      </div>
    </div>
  );
}

const CSS = `
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

.tc-root {
  --bg: #11151B;
  --panel: #171C24;
  --border: #232A35;
  --text: #EDEFF3;
  --muted: #8B93A3;
  --accent: #4FD1B3;
  --pass: #3DDC84;
  --fail: #FF6B6B;
  --warn: #FFB454;
  font-family: 'Space Grotesk', sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100%;
  padding: 28px;
  box-sizing: border-box;
}
.tc-header { display: flex; align-items: center; gap: 8px; margin-bottom: 24px; }
.tc-header-title { font-size: 15px; font-weight: 600; letter-spacing: 0.02em; }
.tc-body { display: flex; gap: 32px; max-width: 720px; margin: 0 auto; }
.tc-rail-wrap { flex-shrink: 0; padding-top: 6px; }
.tc-content { flex: 1; min-width: 0; }

.tc-rail { display: flex; flex-direction: column; }
.tc-rail-item { display: flex; align-items: center; position: relative; min-height: 44px; }
.tc-rail-dot {
  width: 24px; height: 24px; border-radius: 50%; border: 1px solid var(--border);
  display: flex; align-items: center; justify-content: center;
  font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--muted);
  background: var(--panel); flex-shrink: 0; z-index: 1;
}
.tc-rail-dot.active { border-color: var(--accent); color: var(--accent); }
.tc-rail-dot.done { border-color: var(--pass); color: var(--pass); background: rgba(61,220,132,0.08); }
.tc-rail-label { font-size: 12px; color: var(--muted); margin-left: 10px; white-space: nowrap; }
.tc-rail-label.active { color: var(--text); }
.tc-rail-line { position: absolute; left: 11px; top: 24px; width: 1px; height: 44px; background: var(--border); }
.tc-rail-line.done { background: var(--pass); }

.tc-panel { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 28px; }
.tc-eyebrow { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--accent); margin-bottom: 8px; }
.tc-title { font-size: 20px; font-weight: 600; margin: 0 0 14px; }
.tc-lede { color: var(--muted); font-size: 14px; line-height: 1.6; margin: 0 0 22px; max-width: 52ch; }

.tc-choice-grid { display: flex; flex-direction: column; gap: 10px; }
.tc-choice-card {
  display: flex; align-items: center; gap: 16px; text-align: left;
  background: var(--bg); border: 1px solid var(--border); border-radius: 8px;
  padding: 18px 20px; color: var(--text); cursor: pointer; transition: border-color 0.15s;
}
.tc-choice-card:hover { border-color: var(--accent); }
.tc-choice-title { font-size: 15px; font-weight: 600; }
.tc-choice-sub { font-size: 12.5px; color: var(--muted); margin-top: 2px; }
.tc-choice-arrow { margin-left: auto; color: var(--muted); flex-shrink: 0; }

.tc-source-grid { display: flex; gap: 12px; }
.tc-source-card {
  flex: 1; display: flex; flex-direction: column; align-items: flex-start; gap: 6px;
  background: var(--bg); border: 1px dashed var(--border); border-radius: 8px;
  padding: 20px; color: var(--text); cursor: pointer; transition: border-color 0.15s;
}
.tc-source-card:hover { border-color: var(--accent); }
.tc-source-card span:nth-of-type(1) { font-size: 14px; font-weight: 500; margin-top: 4px; }
.tc-source-sub { font-size: 12px; color: var(--muted); }
.tc-hidden-input { display: none; }

.tc-target {
  font-family: 'IBM Plex Mono', monospace; font-size: 12.5px; color: var(--muted);
  background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
  padding: 8px 12px; margin-bottom: 18px; word-break: break-all;
}

.tc-checklist { display: flex; flex-direction: column; gap: 0; margin-bottom: 20px; }
.tc-checkrow { padding: 12px 0; border-bottom: 1px solid var(--border); }
.tc-checkrow:last-child { border-bottom: none; }
.tc-checkrow-top { display: flex; align-items: center; justify-content: space-between; }
.tc-checkrow-label { font-size: 13.5px; }
.tc-checkrow-detail { font-family: 'IBM Plex Mono', monospace; font-size: 11.5px; color: var(--muted); margin-top: 4px; }

.tc-spin { animation: tc-spin 1s linear infinite; }
@keyframes tc-spin { to { transform: rotate(360deg); } }

.tc-banner { border: 1px solid; border-radius: 8px; padding: 16px 18px; margin: 4px 0 18px; }
.tc-banner-top { display: flex; align-items: center; gap: 8px; }
.tc-banner-label { font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 500; letter-spacing: 0.03em; }
.tc-banner-text { font-size: 13px; color: var(--text); opacity: 0.85; margin: 8px 0 0; line-height: 1.5; }

.tc-final-status { font-size: 16px; font-weight: 600; text-align: center; padding-top: 4px; }

.tc-result-grid { display: flex; flex-direction: column; margin-bottom: 16px; }
.tc-result-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 13px 0; border-bottom: 1px solid var(--border); font-size: 14px;
}
.tc-result-row:last-child { border-bottom: none; }

.tc-detail-block { background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: 12px 14px; margin-bottom: 18px; }
.tc-detail-line { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--muted); padding: 3px 0; }
.tc-mono-score { color: var(--accent); }

.tc-footer { display: flex; justify-content: flex-end; margin-top: 6px; gap: 10px; }
.tc-primary-btn, .tc-ghost-btn {
  display: flex; align-items: center; gap: 8px; font-family: 'Space Grotesk', sans-serif;
  font-size: 13.5px; font-weight: 500; border-radius: 6px; padding: 10px 16px; cursor: pointer;
}
.tc-primary-btn { background: var(--accent); border: none; color: #0B1310; }
.tc-ghost-btn { background: transparent; border: 1px solid var(--border); color: var(--muted); }
.tc-ghost-btn:hover { border-color: var(--accent); color: var(--text); }

.tc-error { color: var(--fail); font-size: 12.5px; margin-top: 12px; font-family: 'IBM Plex Mono', monospace; }
`;

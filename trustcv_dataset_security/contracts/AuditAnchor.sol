// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title AuditAnchor — Minimal external audit checkpoint anchor
/// @notice Stores a cryptographic commitment (digest) of a MIRAD audit checkpoint
///         on-chain. Does NOT store the audit log itself.
contract AuditAnchor {
    event AuditAnchored(
        bytes32 indexed auditDigest,
        uint64 sequence,
        bytes32 datasetDigest,
        address indexed submitter,
        uint256 timestamp
    );

    struct Anchor {
        bytes32 auditDigest;
        uint64  sequence;
        bytes32 datasetDigest;
        address submitter;
        uint256 blockNumber;
        uint256 timestamp;
    }

    mapping(bytes32 => Anchor) public anchors;
    bytes32[] public anchorKeys;

    /// @notice Anchor a MIRAD checkpoint digest on-chain.
    /// @param auditDigest   SHA-256 digest of the MIRAD audit checkpoint
    /// @param sequence      Checkpoint sequence number
    /// @param datasetDigest SHA-256 digest of the protected dataset
    function anchor(
        bytes32 auditDigest,
        uint64 sequence,
        bytes32 datasetDigest
    ) external {
        require(auditDigest != bytes32(0), "Empty audit digest");

        anchors[auditDigest] = Anchor({
            auditDigest: auditDigest,
            sequence: sequence,
            datasetDigest: datasetDigest,
            submitter: msg.sender,
            blockNumber: block.number,
            timestamp: block.timestamp
        });
        anchorKeys.push(auditDigest);

        emit AuditAnchored(
            auditDigest,
            sequence,
            datasetDigest,
            msg.sender,
            block.timestamp
        );
    }

    /// @notice Retrieve a stored anchor by its audit digest.
    function getAnchor(bytes32 auditDigest)
        external
        view
        returns (
            uint64  sequence,
            bytes32 datasetDigest,
            address submitter,
            uint256 blockNumber,
            uint256 timestamp
        )
    {
        Anchor storage a = anchors[auditDigest];
        return (a.sequence, a.datasetDigest, a.submitter, a.blockNumber, a.timestamp);
    }

    /// @notice Total number of anchored digests.
    function anchorCount() external view returns (uint256) {
        return anchorKeys.length;
    }
}

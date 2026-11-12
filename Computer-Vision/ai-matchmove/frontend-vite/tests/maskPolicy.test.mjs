import test from "node:test";
import assert from "node:assert/strict";
import { countUnreviewedExclusions } from "../src/maskPolicy.mjs";

test("no regions or null manifests are safe", () => {
  assert.equal(countUnreviewedExclusions(null), 0);
  assert.equal(countUnreviewedExclusions({ frames: [] }), 0);
});

test("background false positive must be reviewed before masked solve", () => {
  const m = {frames: [{regions: [
    {excluded: true, decision: "auto_validated"},
    {excluded: false, decision: "uncertain"},
    {excluded: true, decision: "manual_exclude", origin: "user", prompt: {decision: "exclude"}},
  ]}]};
  assert.equal(countUnreviewedExclusions(m), 1);
});

test("reviewed exclusion no longer blocks solve", () => {
  const m = {frames: [{regions: [{excluded: true, decision: "manual_exclude", review_decision: "exclude"}]}]};
  assert.equal(countUnreviewedExclusions(m), 0);
});

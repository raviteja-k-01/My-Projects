// Safety policy for automatically proposed SAM exclusion regions.
// A candidate is not equivalent to a verified independently moving object.
export function needsExclusionReview(region) {
  return region.excluded && region.review_decision !== "exclude" &&
    !(region.origin === "user" && region.prompt?.decision === "exclude" && !region.review_decision);
}

export function countUnreviewedExclusions(manifest) {
  return (manifest?.frames ?? []).reduce((count, frame) => count + (frame.regions ?? []).filter(
    needsExclusionReview
  ).length, 0);
}

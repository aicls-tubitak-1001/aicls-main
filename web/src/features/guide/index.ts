/**
 * The in-site documentation.
 *
 * The route table mounts {@link GuidePage} three times (one per audience), the workspace renders
 * {@link FirstRunCard}, and the tests reach the section registry and the metric snapshot directly.
 * Nothing else in the application needs to know about the guide's internals.
 */
export { GuidePage, type GuidePageProps } from "./GuidePage";
export { FirstRunCard } from "./FirstRunCard";
export { GUIDE_ROUTES, GUIDE_SEGMENTS } from "./routes";
export {
  GUIDE_SECTIONS,
  STATUS_SECTIONS,
  TECH_SECTIONS,
  USAGE_SECTIONS,
  type GuidePageId,
  type GuideSection,
} from "./sections";
export {
  CONTRACT_TARGETS,
  METRICS_SNAPSHOT,
  formatMetricValue,
  type ContractTarget,
  type GuideMetric,
  type GuideRun,
  type MetricsSnapshot,
} from "./metrics";
export { FIRST_RUN_STORAGE_KEY, isFirstRunVisible, setFirstRunVisible } from "./firstRun";

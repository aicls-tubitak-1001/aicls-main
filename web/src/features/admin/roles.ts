/**
 * Who may enter the administration area.
 *
 * Kept beside the page rather than inside it so the route table and the page can share the list
 * without the page module exporting anything but components (React Fast Refresh stays reliable
 * during a live demo).
 */
import type { Role } from "@/api/types";

/** Roles the backend admits to the administration surface (`StaffUser`). */
export const ADMIN_ROLES: readonly Role[] = ["instructor", "admin"];

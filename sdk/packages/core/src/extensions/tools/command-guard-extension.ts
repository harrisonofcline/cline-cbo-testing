/**
 * Plan-Mode Command Guard Extension
 *
 * Runtime extension that enforces the plan-mode command blacklist
 * (./command-guard.ts) as a `beforeTool` hook. The runtime builder registers
 * it for plan-mode sessions, making command blocking session policy in one
 * shared place: the hook fires for every `run_commands` tool in the runtime —
 * the SDK built-in and host-provided replacements like the VS Code
 * extension's terminal-backed tool — without threading a flag through each
 * layer.
 *
 * Because `beforeTool` hooks run before tool policies and user approval,
 * a blocked call is rejected up front: the user is never asked to approve a
 * command that would only fail, and the model receives the plan-mode error
 * as the tool result (`skip`, not `stop`, so the run continues).
 */

import type {
	AgentBeforeToolContext,
	AgentBeforeToolResult,
	AgentExtension,
	AgentMessage,
	AgentMode,
	ITelemetryService,
} from "@cline/shared";
import { parseUserInputMode } from "@cline/shared";
import { capturePlanModeCommandBlocked } from "../../services/telemetry/core-events";
import {
	findFileEditingCommand,
	formatPlanModeBlockedCommandError,
} from "./command-guard";
import { DefaultToolNames } from "./constants";
import { normalizeRunCommandsInput } from "./helpers";

export const PLAN_MODE_COMMAND_GUARD_EXTENSION_NAME =
	"core.plan-mode-command-guard";

export const PLAN_MODE_GUARD_EXTENSION_NAME = "core.plan-mode-guard";

const PLAN_MODE_WRITE_TOOL_NAMES = new Set<string>([
	DefaultToolNames.EDITOR,
	DefaultToolNames.APPLY_PATCH,
]);

function inferActiveModeFromMessages(
	messages: readonly AgentMessage[] | undefined,
): AgentMode | undefined {
	if (!messages) return undefined;
	for (let i = messages.length - 1; i >= 0; i--) {
		const msg = messages[i];
		if (!msg || msg.role !== "user") continue;
		const content = msg.content;
		const text =
			typeof content === "string"
				? content
				: Array.isArray(content)
					? content
							.map((part) =>
								part && typeof part === "object" && part.type === "text"
									? part.text
									: "",
							)
							.join("\n")
					: "";
		const parsed = parseUserInputMode(text);
		if (parsed) return parsed;
	}
	return undefined;
}

export interface PlanModeCommandGuardOptions {
	telemetry?: ITelemetryService;
	/**
	 * Optional dynamic mode getter for `prefixStableModes` sessions where the
	 * session stays alive across Plan <-> Act transitions without rebuilding
	 * its tools or system prompt. When omitted and `blockWriteTools` is false,
	 * the guard assumes it was registered specifically for a plan-mode runtime.
	 */
	getMode?: () => AgentMode | undefined;
	/**
	 * When true, also block non-read-only mutating tools (`editor`, `apply_patch`,
	 * or any tool explicitly flagged `readOnly === false`) while active mode is
	 * `"plan"`. Used by `prefixStableModes` where Act and Plan share one toolset.
	 */
	blockWriteTools?: boolean;
}

export function createPlanModeCommandGuardExtension(
	options: PlanModeCommandGuardOptions = {},
): AgentExtension {
	const isPlanModeActive = (context: AgentBeforeToolContext): boolean => {
		if (options.getMode) {
			const liveMode = options.getMode();
			if (liveMode) return liveMode === "plan";
		}
		if (options.blockWriteTools) {
			const msgMode = inferActiveModeFromMessages(context.snapshot.messages);
			return msgMode === "plan";
		}
		return true;
	};

	const beforeTool = (
		context: AgentBeforeToolContext,
	): AgentBeforeToolResult | undefined => {
		if (!isPlanModeActive(context)) {
			return undefined;
		}

		if (options.blockWriteTools) {
			const tool = context.tool as { name: string; readOnly?: boolean };
			const isExplicitWriteTool =
				PLAN_MODE_WRITE_TOOL_NAMES.has(tool.name) || tool.readOnly === false;
			if (isExplicitWriteTool && tool.readOnly !== true) {
				return {
					skip: true,
					reason: `Tool "${tool.name}" is disabled in Plan mode. Complete exploration and present your plan first; file edits happen in Act mode.`,
				};
			}
		}

		if (context.tool.name !== DefaultToolNames.RUN_COMMANDS) {
			return undefined;
		}

		let commands: ReturnType<typeof normalizeRunCommandsInput>;
		try {
			commands = normalizeRunCommandsInput(context.input);
		} catch {
			// Unparseable input: let the tool produce its own validation error.
			return undefined;
		}

		for (const command of commands) {
			const blocked = findFileEditingCommand(command);
			if (blocked) {
				capturePlanModeCommandBlocked(options.telemetry, {
					tool_name: "run_commands",
					blocked_construct: blocked,
					command_count: commands.length,
					agent_id: context.snapshot.agentId,
					conversation_id: context.snapshot.conversationId,
					run_id: context.snapshot.runId,
					iteration: context.snapshot.iteration,
					tool_call_id: context.toolCall.toolCallId,
				});
				return {
					skip: true,
					reason: formatPlanModeBlockedCommandError(blocked),
				};
			}
		}

		return undefined;
	};

	return {
		name: options.blockWriteTools
			? PLAN_MODE_GUARD_EXTENSION_NAME
			: PLAN_MODE_COMMAND_GUARD_EXTENSION_NAME,
		manifest: {
			capabilities: ["hooks"],
		},
		hooks: {
			beforeTool,
		},
	};
}

/**
 * Dynamic Plan-mode guard for `prefixStableModes` sessions: keeps the full
 * Act-mode tool schema registered (zero cache invalidation on mode switches)
 * while dynamically rejecting mutating tools and file-editing shell commands
 * whenever the active mode is `"plan"`.
 */
export function createPlanModeGuardExtension(
	options: Omit<PlanModeCommandGuardOptions, "blockWriteTools"> = {},
): AgentExtension {
	return createPlanModeCommandGuardExtension({
		...options,
		blockWriteTools: true,
	});
}

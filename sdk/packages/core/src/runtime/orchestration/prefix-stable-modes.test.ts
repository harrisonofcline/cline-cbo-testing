import {
	type AgentBeforeToolContext,
	buildClineSystemPrompt,
	formatUserInputBlock,
} from "@cline/shared";
import { describe, expect, it } from "vitest";
import {
	createPlanModeGuardExtension,
	findFileEditingCommand,
	PLAN_MODE_GUARD_EXTENSION_NAME,
} from "../../extensions/tools";
import { DefaultRuntimeBuilder } from "./runtime-builder";

function makeToolContext(
	toolName: string,
	input: unknown,
	options: { readOnly?: boolean; userMode?: "plan" | "act" } = {},
): AgentBeforeToolContext {
	return {
		tool: {
			name: toolName,
			description: "test",
			inputSchema: { type: "object" },
			readOnly: options.readOnly,
			execute: async () => "ok",
		} as AgentBeforeToolContext["tool"],
		input,
		toolCall: { toolCallId: "tc-1", toolName, input },
		snapshot: {
			agentId: "agent-1",
			conversationId: "conv-1",
			runId: "run-1",
			iteration: 1,
			messages: options.userMode
				? [
						{
							id: "m1",
							role: "user",
							content: [
								{
									type: "text",
									text: formatUserInputBlock("do work", options.userMode),
								},
							],
							createdAt: 1,
						},
					]
				: [],
		} as unknown as AgentBeforeToolContext["snapshot"],
	};
}

describe("prefixStableModes Plan <-> Act wire parity", () => {
	it("produces byte-identical system prompts for Plan and Act when prefixStableModes is true", () => {
		const planPrompt = buildClineSystemPrompt({
			workspaceRoot: "/tmp/repo",
			mode: "plan",
			prefixStableModes: true,
		});
		const actPrompt = buildClineSystemPrompt({
			workspaceRoot: "/tmp/repo",
			mode: "act",
			prefixStableModes: true,
		});
		expect(planPrompt).toBe(actPrompt);
	});

	it("registers identical tool names in Plan and Act and dynamically blocks writes only in Plan mode", async () => {
		const builder = new DefaultRuntimeBuilder();
		const baseConfig = {
			providerId: "anthropic",
			modelId: "claude-sonnet-5-5",
			cwd: process.cwd(),
			enableTools: true,
			enableSpawnAgent: false,
			enableAgentTeams: false,
			disableMcpSettingsTools: true,
			prefixStableModes: true,
		};

		const planEnv = await builder.build({
			config: { ...baseConfig, mode: "plan" },
		});
		const actEnv = await builder.build({
			config: { ...baseConfig, mode: "act" },
		});

		const planToolNames = planEnv.tools.map((t) => t.name);
		const actToolNames = actEnv.tools.map((t) => t.name);
		expect(planToolNames).toEqual(actToolNames);
		expect(planToolNames).toContain("editor");

		const guard = planEnv.extensions?.find(
			(ext) => ext.name === PLAN_MODE_GUARD_EXTENSION_NAME,
		);
		expect(guard).toBeDefined();

		// Dynamic mode toggling via getMode / <user_input mode="...">
		let liveMode: "plan" | "act" = "plan";
		const dynamicGuard = createPlanModeGuardExtension({
			getMode: () => liveMode,
		});
		const hook = dynamicGuard.hooks?.beforeTool;
		expect(hook).toBeDefined();

		expect(
			await hook?.(makeToolContext("editor", { path: "a.ts" })),
		).toMatchObject({ skip: true });
		expect(
			await hook?.(
				makeToolContext("run_commands", { commands: ["rm -rf tmp"] }),
			),
		).toMatchObject({ skip: true });
		expect(
			await hook?.(
				makeToolContext("run_commands", { commands: ["git status"] }),
			),
		).toBeUndefined();
		expect(
			await hook?.(makeToolContext("custom_write", {}, { readOnly: false })),
		).toMatchObject({ skip: true });

		// Switch to Act in place: all writes now allowed without rebuilding tools!
		liveMode = "act";
		expect(
			await hook?.(makeToolContext("editor", { path: "a.ts" })),
		).toBeUndefined();
		expect(
			await hook?.(
				makeToolContext("run_commands", { commands: ["rm -rf tmp"] }),
			),
		).toBeUndefined();

		// Fallback inference from <user_input mode="plan"> when getMode is omitted
		const messageInferredGuard = createPlanModeGuardExtension();
		expect(
			await messageInferredGuard.hooks?.beforeTool?.(
				makeToolContext("editor", {}, { userMode: "plan" }),
			),
		).toMatchObject({ skip: true });
		expect(
			await messageInferredGuard.hooks?.beforeTool?.(
				makeToolContext("editor", {}, { userMode: "act" }),
			),
		).toBeUndefined();

		// Exported helper works for downstream consumers
		expect(findFileEditingCommand("sed -i 's/a/b/' file.ts")).toBeDefined();
		expect(findFileEditingCommand("git diff")).toBeUndefined();
	});
});

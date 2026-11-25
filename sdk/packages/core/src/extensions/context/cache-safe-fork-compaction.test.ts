import { createHandlerAsync } from "@cline/llms";
import type { Message, ToolDefinition } from "@cline/shared";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { CoreCompactionContext } from "../../types/config";
import { runAgenticCompaction } from "./agentic-compaction";
import { summarizeConversation } from "./summarize-conversation";

vi.mock("@cline/llms", async (importOriginal) => {
	const actual = await importOriginal<typeof import("@cline/llms")>();
	return { ...actual, createHandlerAsync: vi.fn() };
});

const mockedCreateHandlerAsync = vi.mocked(createHandlerAsync);

function makeContext(
	apiMessages: Message[],
	systemPrompt: string,
	tools: ToolDefinition[],
): CoreCompactionContext {
	return {
		agentId: "agent-1",
		conversationId: "conv-1",
		parentAgentId: null,
		iteration: 4,
		messages: apiMessages.map((m, idx) => ({ ...m, ts: idx + 1 })),
		apiMessages: apiMessages.map((m, idx) => ({ ...m, ts: idx + 1 })),
		systemPrompt,
		tools,
		model: { id: "claude-sonnet-5-5", provider: "anthropic" },
		mode: "auto",
		budget: {
			request: {
				inputTokens: 100_000,
				maxInputTokens: 200_000,
				triggerTokens: 140_000,
				targetTokens: 80_000,
				overheadTokens: 4_000,
				thresholdRatio: 0.7,
				utilizationRatio: 0.5,
			},
			messages: {
				inputTokens: 96_000,
				triggerTokens: 136_000,
				targetTokens: 76_000,
			},
		},
	};
}

describe("cache-safe fork compaction", () => {
	beforeEach(() => {
		mockedCreateHandlerAsync.mockReset();
	});

	it("preserves exact systemPrompt, tools, and apiMessages prefix", async () => {
		const calls: Array<{
			systemPrompt: string;
			messages: Message[];
			tools?: ToolDefinition[];
		}> = [];
		mockedCreateHandlerAsync.mockResolvedValue({
			createMessage: (
				systemPrompt: string,
				messages: Message[],
				tools?: ToolDefinition[],
			) => {
				calls.push({ systemPrompt, messages, tools });
				return (async function* () {
					yield { type: "text" as const, id: "1", text: "Goal: ship fork." };
					yield { type: "done" as const, id: "2", success: true };
				})();
			},
			getModel: () => ({
				id: "claude-sonnet-5-5",
				info: { id: "claude-sonnet-5-5" },
			}),
		} as never);

		const apiMessages: Message[] = [
			{ role: "user", content: "Inspect the codebase." },
			{ role: "assistant", content: "Inspected packages/core." },
			{ role: "user", content: "Proceed." },
		];
		const tools: ToolDefinition[] = [
			{
				name: "read_file",
				description: "Read",
				inputSchema: { type: "object" },
			},
		];
		const context = makeContext(apiMessages, "Exact cached prompt.", tools);

		const result = await runAgenticCompaction({
			context,
			providerConfig: { providerId: "anthropic", modelId: "claude-sonnet-5-5" },
			cacheSafeFork: true,
			preserveRecentTokens: 10,
			estimateMessageTokens: () => 50,
		});

		expect(result).toBeDefined();
		expect(calls).toHaveLength(1);
		expect(calls[0]?.systemPrompt).toBe("Exact cached prompt.");
		expect(calls[0]?.tools).toEqual(tools);
		expect(calls[0]?.messages.slice(0, apiMessages.length)).toEqual(
			context.apiMessages,
		);
		expect(calls[0]?.messages).toHaveLength(apiMessages.length + 1);
	});

	it("falls back to cold summarizer when fork attempts tool calls", async () => {
		const calls: Array<{ systemPrompt: string; messageCount: number }> = [];
		let attempt = 0;
		mockedCreateHandlerAsync.mockResolvedValue({
			createMessage: (systemPrompt: string, messages: Message[]) => {
				calls.push({ systemPrompt, messageCount: messages.length });
				attempt += 1;
				if (attempt === 1) {
					return (async function* () {
						yield {
							type: "tool_calls" as const,
							id: "1",
							tool_call: {
								call_id: "tc-1",
								function: { id: "tc-1", name: "read_file", arguments: "{}" },
							},
						};
						yield { type: "done" as const, id: "2", success: true };
					})();
				}
				return (async function* () {
					yield { type: "text" as const, id: "3", text: "Fallback summary." };
					yield { type: "done" as const, id: "4", success: true };
				})();
			},
			getModel: () => ({
				id: "claude-sonnet-5-5",
				info: { id: "claude-sonnet-5-5" },
			}),
		} as never);

		const context = makeContext(
			[
				{ role: "user", content: "Start" },
				{ role: "assistant", content: "Working" },
				{ role: "user", content: "Continue" },
			],
			"Active prompt",
			[
				{
					name: "read_file",
					description: "Read",
					inputSchema: { type: "object" },
				},
			],
		);

		const result = await runAgenticCompaction({
			context,
			providerConfig: { providerId: "anthropic", modelId: "claude-sonnet-5-5" },
			cacheSafeFork: true,
			preserveRecentTokens: 10,
			estimateMessageTokens: () => 50,
		});

		expect(result).toBeDefined();
		expect(calls).toHaveLength(2);
		expect(calls[1]?.messageCount).toBe(1);
	});

	it("supports cacheSafeFork in summarizeConversation", async () => {
		const calls: Array<{ systemPrompt: string; messages: Message[] }> = [];
		mockedCreateHandlerAsync.mockResolvedValue({
			createMessage: (systemPrompt: string, messages: Message[]) => {
				calls.push({ systemPrompt, messages });
				return (async function* () {
					yield { type: "text" as const, id: "1", text: "Warm fork note." };
					yield { type: "done" as const, id: "2", success: true };
				})();
			},
			getModel: () => ({
				id: "claude-sonnet-5-5",
				info: { id: "claude-sonnet-5-5" },
			}),
		} as never);

		const apiMessages = [
			{ role: "user" as const, content: "Build feature X", ts: 1 },
			{ role: "assistant" as const, content: "Done step 1", ts: 2 },
		];
		const note = await summarizeConversation({
			messages: apiMessages,
			apiMessages,
			systemPrompt: "Warm trunk prompt",
			cacheSafeFork: true,
			config: { providerId: "anthropic", modelId: "claude-sonnet-5-5" },
		});

		expect(note).toBe("Warm fork note.");
		expect(calls).toHaveLength(1);
		expect(calls[0]?.systemPrompt).toBe("Warm trunk prompt");
		expect(calls[0]?.messages).toHaveLength(3);
	});
});

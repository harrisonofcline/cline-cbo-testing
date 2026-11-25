import type {
	AgentModelEvent,
	GatewayProviderContext,
	GatewayStreamRequest,
} from "@cline/shared";
import { describe, expect, it } from "vitest";
import { createGateway } from "./gateway";
import { composeAiSdkProviderOptions } from "./routing/provider-options";

describe("structured GatewayPromptCacheRequest controls", () => {
	it("composes OpenAI v2 opt-out, legacy 24h retention, Vercel caching:auto, and Fireworks allowed_tools", () => {
		const dummyCtx = (
			providerId: string,
			modelId: string,
		): GatewayProviderContext => ({
			provider: {
				id: providerId,
				name: providerId,
				defaultModelId: modelId,
				models: [],
			},
			model: { id: modelId, name: modelId, providerId },
			config: { providerId },
		});

		const oaiV2 = composeAiSdkProviderOptions(
			{
				providerId: "openai",
				modelId: "gpt-6.2-sol",
				messages: [],
				promptCache: { mode: "off", cohortKey: "cohort-1" },
			},
			dummyCtx("openai", "gpt-6.2-sol"),
		);
		expect(oaiV2.openai).toMatchObject({
			prompt_cache_key: "cohort-1",
			prompt_cache_options: { mode: "explicit" },
		});

		const oaiLegacy = composeAiSdkProviderOptions(
			{
				providerId: "openai",
				modelId: "gpt-5.5",
				messages: [],
				promptCache: { ttl: "24h", cohortKey: "cohort-2" },
			},
			dummyCtx("openai", "gpt-5.5"),
		);
		expect(oaiLegacy.openai).toMatchObject({
			prompt_cache_key: "cohort-2",
			prompt_cache_retention: "24h",
		});

		const vercel = composeAiSdkProviderOptions(
			{
				providerId: "vercel-ai-gateway",
				modelId: "anthropic/claude-sonnet-5.5",
				messages: [],
				promptCache: { mode: "auto", cohortKey: "cohort-v" },
			},
			dummyCtx("vercel-ai-gateway", "anthropic/claude-sonnet-5.5"),
		);
		expect(vercel.gateway).toEqual({ caching: "auto" });

		const fw = composeAiSdkProviderOptions(
			{
				providerId: "fireworks",
				modelId: "accounts/fireworks/models/qwen-4-max",
				messages: [],
				promptCache: { allowedTools: ["read_files", "search_codebase"] },
			},
			dummyCtx("fireworks", "accounts/fireworks/models/qwen-4-max"),
		);
		expect(fw.fireworks).toMatchObject({
			tool_choice: {
				type: "allowed_tools",
				mode: "auto",
				tools: [
					{ type: "function", function: { name: "read_files" } },
					{ type: "function", function: { name: "search_codebase" } },
				],
			},
		});
	});

	it("forwards AgentModelRequest.options.promptCache through GatewayModelAdapter and injects multi-binding Fireworks/Vercel/xAI sticky headers", async () => {
		const seenHeaders: Array<{ url: string; headers: Headers; body: string }> =
			[];
		const fakeFetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
			const url =
				typeof input === "string"
					? input
					: input instanceof URL
						? input.href
						: input.url;
			const headers = new Headers(
				input instanceof Request ? input.headers : undefined,
			);
			new Headers(init?.headers).forEach((v, k) => {
				headers.set(k, v);
			});
			const body =
				typeof init?.body === "string"
					? init.body
					: input instanceof Request
						? await input.clone().text()
						: "";
			seenHeaders.push({ url, headers, body });
			const sse = [
				`data: ${JSON.stringify({ id: "chatcmpl-1", object: "chat.completion.chunk", created: 1, model: "test", choices: [{ index: 0, delta: { content: "ok" }, finish_reason: null }] })}\n\n`,
				`data: ${JSON.stringify({ id: "chatcmpl-1", object: "chat.completion.chunk", created: 1, model: "test", choices: [{ index: 0, delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 } })}\n\n`,
				"data: [DONE]\n\n",
			].join("");
			return new Response(sse, {
				status: 200,
				headers: { "content-type": "text/event-stream" },
			});
		}) as typeof fetch;

		const gateway = createGateway({
			fetch: fakeFetch,
			providerConfigs: [
				{ providerId: "fireworks", apiKey: "fw-test" },
				{ providerId: "vercel-ai-gateway", apiKey: "vg-test" },
				{ providerId: "xai", apiKey: "xai-test" },
			],
		});

		const fwModel = gateway.createAgentModel({
			providerId: "fireworks",
			modelId: "accounts/fireworks/models/deepseek-v4-1-flash",
		});
		const fwEvents: AgentModelEvent[] = [];
		for await (const ev of await fwModel.stream({
			messages: [
				{
					id: "u1",
					role: "user",
					content: [{ type: "text", text: "hi" }],
					createdAt: 1,
				},
			],
			tools: [],
			options: {
				metadata: { sessionId: "sess-1" },
				promptCache: {
					cohortKey: "wave-42",
					isolationKey: "tenant-abc",
				},
			},
		})) {
			fwEvents.push(ev);
		}
		expect(seenHeaders[0]?.headers.get("x-session-affinity")).toBe("wave-42");
		expect(seenHeaders[0]?.headers.get("x-multi-turn-session-id")).toBe(
			"wave-42",
		);
		expect(seenHeaders[0]?.headers.get("x-prompt-cache-isolation-key")).toBe(
			"tenant-abc",
		);

		const vgStream = await gateway.stream({
			providerId: "vercel-ai-gateway",
			modelId: "anthropic/claude-sonnet-5.5",
			messages: [
				{
					id: "u1",
					role: "user",
					content: [{ type: "text", text: "hi" }],
					createdAt: 1,
				},
			],
			promptCache: { cohortKey: "vg-cohort" },
		} satisfies GatewayStreamRequest);
		for await (const _ev of vgStream) {
			// drain
		}
		expect(seenHeaders[1]?.headers.get("x-session-id")).toBe("vg-cohort");

		const xaiStream = await gateway.stream({
			providerId: "xai",
			modelId: "grok-4.20-0309-non-reasoning",
			messages: [
				{
					id: "u1",
					role: "user",
					content: [{ type: "text", text: "hi" }],
					createdAt: 1,
				},
			],
			metadata: { sessionId: "grok-s1" },
		});
		for await (const _ev of xaiStream) {
			// drain
		}
		expect(seenHeaders[2]?.headers.get("x-grok-conv-id")).toBe("grok-s1");
	});
});

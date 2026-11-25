import { createHandlerAsync } from "@cline/llms";
import type { Message, ToolDefinition } from "@cline/shared";
import type { ProviderConfig } from "../../types/provider-settings";

async function collectSummaryStream(
	stream: AsyncIterable<import("@cline/llms").ApiStreamChunk>,
	options: { maxOutputChars?: number; signal?: AbortSignal },
) {
	let text = "";
	let reasoningChars = 0;
	let failed = false;
	let usedTools = false;
	let incompleteReason: string | undefined;
	for await (const chunk of stream) {
		options.signal?.throwIfAborted();
		if (chunk.type === "text") {
			text += chunk.text;
			if (options.maxOutputChars && text.length > options.maxOutputChars) {
				throw new Error("Summary exceeded its output limit.");
			}
		} else if (chunk.type === "reasoning") {
			reasoningChars += chunk.reasoning?.length ?? 0;
		} else if (chunk.type === "done") {
			if (!chunk.success && chunk.error) throw new Error(chunk.error);
			incompleteReason = chunk.incompleteReason ?? incompleteReason;
			failed ||= !chunk.success;
		} else if (chunk.type === "tool_calls") {
			usedTools = true;
		}
	}
	options.signal?.throwIfAborted();
	// Handlers may finish normally without emitting an optional done chunk.
	return { text, reasoningChars, failed, usedTools, incompleteReason };
}

export async function generateSummary(options: {
	providerConfig: ProviderConfig;
	systemPrompt: string;
	request: string;
	maxOutputChars?: number;
	signal?: AbortSignal;
}) {
	options.signal?.throwIfAborted();
	const handler = await createHandlerAsync(
		options.signal
			? { ...options.providerConfig, abortSignal: options.signal }
			: options.providerConfig,
	);
	options.signal?.throwIfAborted();
	return collectSummaryStream(
		handler.createMessage(options.systemPrompt, [
			{ role: "user", content: options.request },
		]),
		options,
	);
}

/**
 * Generates a compaction summary by appending a single summarization user turn
 * onto the active session's exact `systemPrompt`, `tools`, and `prefixMessages`
 * so the provider serves the warm trunk from KV cache instead of cold-prefilling.
 */
export async function generateCacheSafeForkSummary(options: {
	providerConfig: ProviderConfig;
	systemPrompt: string;
	tools?: ToolDefinition[];
	prefixMessages: Message[];
	summaryPrompt: string;
	maxOutputChars?: number;
	signal?: AbortSignal;
}) {
	options.signal?.throwIfAborted();
	const handler = await createHandlerAsync(
		options.signal
			? { ...options.providerConfig, abortSignal: options.signal }
			: options.providerConfig,
	);
	options.signal?.throwIfAborted();
	return collectSummaryStream(
		handler.createMessage(
			options.systemPrompt,
			[
				...options.prefixMessages,
				{ role: "user", content: options.summaryPrompt },
			],
			options.tools,
		),
		options,
	);
}

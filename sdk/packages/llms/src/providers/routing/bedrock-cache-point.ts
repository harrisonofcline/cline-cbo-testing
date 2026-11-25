import type {
	GatewayModelRoute,
	GatewayProviderContext,
	GatewayProviderMetadata,
	GatewayStreamRequest,
} from "@cline/shared";
import { modelRouteMatches, resolveModelFamily } from "../model-facts";

/**
 * Bedrock's Converse API expresses prompt-cache checkpoints as dedicated
 * `cachePoint` content blocks. `@ai-sdk/amazon-bedrock` only emits them from
 * `providerOptions.bedrock.cachePoint` markers and silently drops the
 * Anthropic `cache_control` dialect, so Bedrock needs its own prompt-cache
 * wire format.
 */
export const BEDROCK_ROUTING_METADATA: GatewayProviderMetadata = {
	routing: {
		promptCache: {
			format: "bedrock-cache-point",
			routes: [{ matcher: "anthropic-compatible" }],
		},
		reasoning: {
			format: "anthropic-thinking",
			routes: [{ matcher: "anthropic-compatible" }],
		},
	},
};

export function resolveBedrockCachePointRoute(
	request: GatewayStreamRequest,
	context: GatewayProviderContext,
): GatewayModelRoute | undefined {
	const promptCache = context.provider.metadata?.routing?.promptCache;
	if (promptCache?.format !== "bedrock-cache-point") {
		return undefined;
	}

	return promptCache.routes.find((route) =>
		modelRouteMatches(route, {
			modelId: request.modelId,
			family: resolveModelFamily(context),
			capabilities: context.model.capabilities,
		}),
	);
}

export function shouldApplyBedrockCachePoint(
	request: GatewayStreamRequest,
	context: GatewayProviderContext,
): boolean {
	if (request.promptCache?.mode === "off") {
		return false;
	}
	return resolveBedrockCachePointRoute(request, context) !== undefined;
}

export function createBedrockCachePointProviderOptions(ttl?: "5m" | "1h") {
	return {
		bedrock: {
			cachePoint:
				ttl === "1h"
					? ({ type: "default", ttl: "1h" } as const)
					: ({ type: "default" } as const),
		},
	};
}

const MAX_BEDROCK_CACHE_POINTS = 4;

/**
 * Attach Bedrock cache points at explicit message indices (deduped, sorted,
 * capped at 4).
 */
export function applyBedrockCachePoints(
	messages: Array<Record<string, unknown>>,
	indices: readonly number[],
	ttl?: "5m" | "1h",
): void {
	const valid = [
		...new Set(
			indices
				.map((idx) => Math.trunc(idx))
				.filter((idx) => idx >= 0 && idx < messages.length),
		),
	]
		.sort((a, b) => a - b)
		.slice(0, MAX_BEDROCK_CACHE_POINTS);
	const marker = createBedrockCachePointProviderOptions(ttl);
	for (const idx of valid) {
		const message = messages[idx];
		if (!message) continue;
		message.providerOptions = {
			...(message.providerOptions as Record<string, unknown> | undefined),
			...marker,
		};
	}
}

/**
 * Attach a message-level cache-point marker to the last user message. The
 * Bedrock message converter appends the `cachePoint` block after that
 * message's content, caching the whole prefix (tools, system, and history)
 * up to it — mirroring the Anthropic writer's checkpoint placement.
 */
export function applyBedrockCachePointToLastUserMessage(
	messages: Array<Record<string, unknown>>,
	ttl?: "5m" | "1h",
): void {
	for (let i = messages.length - 1; i >= 0; i--) {
		const message = messages[i];
		if (message?.role !== "user") {
			continue;
		}
		message.providerOptions = {
			...(message.providerOptions as Record<string, unknown> | undefined),
			...createBedrockCachePointProviderOptions(ttl),
		};
		return;
	}
}

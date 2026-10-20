/* Licensed under MIT 2025-2026. */
package edu.kit.kastel.sdq.lissa.ratlr.classifier;

import static dev.langchain4j.internal.Utils.quoted;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.jspecify.annotations.Nullable;

import edu.kit.kastel.sdq.lissa.ratlr.cache.Cache;
import edu.kit.kastel.sdq.lissa.ratlr.cache.CacheManager;
import edu.kit.kastel.sdq.lissa.ratlr.cache.classifier.ClassifierCacheKey;
import edu.kit.kastel.sdq.lissa.ratlr.configuration.ModuleConfiguration;
import edu.kit.kastel.sdq.lissa.ratlr.context.ContextStore;
import edu.kit.kastel.sdq.lissa.ratlr.knowledge.Element;

import dev.langchain4j.data.message.*;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.response.ChatResponse;

/**
 * A classifier that uses a language model to reason about trace links between elements.
 * This classifier employs a chat-based approach to determine if elements are related,
 * using configurable prompts and caching to improve performance.
 */
public class ReasoningClassifier extends Classifier {
    private static final int MAX_RESPONSE_ATTEMPTS = 3;

    /**
     * The identifier used for this classifier type in configuration and caching.
     */
    public static final String REASONING_CLASSIFIER_NAME = "reasoning";

    /**
     * The configuration key for the classification prompt.
     */
    private static final String CLASSIFICATION_PROMPT_KEY = "prompt";

    private final Cache<ClassifierCacheKey> cache;

    /**
     * Provider for the language model used in classification.
     */
    private final ChatLanguageModelProvider provider;

    /**
     * The language model instance used for classification.
     */
    private final ChatModel llm;

    /**
     * The prompt template used for classification requests.
     */
    private String prompt;

    /**
     * Whether to use original artifacts instead of nested elements.
     */
    private final boolean useOriginalArtifacts;

    /**
     * Whether to include a system message in the chat.
     */
    private final boolean useSystemMessage;

    /** Whether accepted links must also be classified as Refine or Satisfy. */
    private final boolean classifyRelationType;

    /**
     * Creates a new reasoning classifier with the specified configuration.
     *
     * @param configuration The module configuration containing classifier settings
     * @param contextStore The shared context store for pipeline components
     */
    public ReasoningClassifier(ModuleConfiguration configuration, ContextStore contextStore) {
        super(ChatLanguageModelProvider.threads(configuration), contextStore);
        this.provider = new ChatLanguageModelProvider(configuration);
        this.cache = CacheManager.getDefaultInstance().getCache(this, provider.cacheParameters());
        this.prompt = configuration.argumentAsStringByEnumIndex(
                CLASSIFICATION_PROMPT_KEY,
                0,
                ReasoningClassifierPrompt.values(),
                ReasoningClassifierPrompt::getPromptTemplate);
        this.useOriginalArtifacts = configuration.argumentAsBoolean("use_original_artifacts", false);
        this.useSystemMessage = configuration.argumentAsBoolean("use_system_message", true);
        this.classifyRelationType = configuration.argumentAsBoolean("classify_relation_type", false);
        this.llm = this.provider.createChatModel();
    }

    /**
     * Creates a new reasoning classifier with the specified parameters.
     * This constructor is used internally for creating thread-local copies.
     *
     * @param threads The number of threads to use for parallel processing
     * @param cache The cache to use for storing classification results
     * @param provider The language model provider
     * @param prompt The prompt template to use
     * @param useOriginalArtifacts Whether to use original artifacts
     * @param useSystemMessage Whether to include a system message
     */
    private ReasoningClassifier(
            int threads,
            Cache<ClassifierCacheKey> cache,
            ChatLanguageModelProvider provider,
            String prompt,
            boolean useOriginalArtifacts,
            boolean useSystemMessage,
            boolean classifyRelationType,
            ContextStore contextStore) {
        super(threads, contextStore);
        this.cache = cache;
        this.provider = provider;
        this.prompt = prompt;
        this.useOriginalArtifacts = useOriginalArtifacts;
        this.useSystemMessage = useSystemMessage;
        this.classifyRelationType = classifyRelationType;
        this.llm = this.provider.createChatModel();
    }

    @Override
    protected final Classifier copyOf() {
        return new ReasoningClassifier(
                threads,
                cache,
                provider,
                prompt,
                useOriginalArtifacts,
                useSystemMessage,
                classifyRelationType,
                contextStore);
    }

    @Override
    public void setClassificationPrompt(String prompt) {
        this.prompt = prompt;
    }

    public static String getClassificationPromptKey() {
        return CLASSIFICATION_PROMPT_KEY;
    }

    /**
     * Classifies a pair of elements by using the language model to reason about their relationship.
     * The classification result is cached to avoid redundant LLM calls.
     *
     * @param source The source element
     * @param target The target element
     * @return A classification result if the elements are related, empty otherwise
     */
    @Override
    protected final Optional<ClassificationResult> classify(Element source, Element target) {
        var targetToConsider = target;
        if (useOriginalArtifacts) {
            while (targetToConsider.getParent() != null) {
                targetToConsider = targetToConsider.getParent();
            }
        }

        var sourceToConsider = source;
        /* TODO Maybe reactivate the sourceToConsider in the future ..
        if(useOriginalArtifacts){
            while (sourceToConsider.getParent() != null) {
                sourceToConsider = sourceToConsider.getParent();
            }
        }
        */

        String llmResponse = classifyIntern(sourceToConsider, targetToConsider);
        TraceDecision decision = classifyRelationType
                ? parseTraceDecision(llmResponse)
                : new TraceDecision(Pattern.compile("<trace>\\s*yes\\s*</trace>", Pattern.CASE_INSENSITIVE)
                        .matcher(llmResponse).find(), "None");
        if (classifyRelationType) {
            logger.info(
                    "TYPED_TRACE_DECISION\t{}\t{}\t{}\t{}",
                    source.getIdentifier(),
                    targetToConsider.getIdentifier(),
                    decision.related() ? "yes" : "no",
                    decision.relationType());
        }
        if (decision.related()) {
            return Optional.of(ClassificationResult.of(source, targetToConsider));
        }
        return Optional.empty();
    }

    /**
     * Determines if the language model's response indicates a trace link.
     * The response is expected to contain a trace tag with "yes" or "no".
     *
     * @param llmResponse The response from the language model
     * @return true if the response indicates a trace link, false otherwise
     */
    static TraceDecision parseTraceDecision(String llmResponse) {
        // Anchor to the final adjacent pair: reasoning may contain example tags.
        Matcher match = Pattern.compile(
                        "<trace>\\s*(yes|no)\\s*</trace>\\s*<relation_type>\\s*(Refine|Satisfy|None)\\s*</relation_type>\\s*\\z",
                        Pattern.CASE_INSENSITIVE)
                .matcher(llmResponse);
        if (!match.find()) {
            throw new IllegalArgumentException("Missing valid final trace/relation_type tag pair");
        }
        boolean related = match.group(1).equalsIgnoreCase("yes");
        String relation = match.group(2).toLowerCase(Locale.ROOT);
        if (related == relation.equals("none")) {
            throw new IllegalArgumentException("Inconsistent final trace and relation_type values");
        }
        return new TraceDecision(related, switch (relation) {
            case "refine" -> "Refine";
            case "satisfy" -> "Satisfy";
            default -> "None";
        });
    }

    /**
     * Performs the actual classification using the language model.
     * The result is cached to avoid redundant LLM calls.
     *
     * @param source The source element
     * @param target The target element
     * @return The language model's response
     */
    private String classifyIntern(Element source, Element target) {
        List<ChatMessage> messages = new ArrayList<>();
        if (useSystemMessage) {
            String systemMessage = classifyRelationType
                    ? "Your job is to determine whether two system artifacts have a traceability link and, when they do, classify that link as Refine or Satisfy."
                    : "Your job is to determine if there is a traceability link between two artifacts of a system.";
            messages.add(new SystemMessage(systemMessage));
        }

        String request = prompt.replace("{source_type}", source.getType())
                .replace("{source_content}", source.getContent())
                .replace("{target_type}", target.getType())
                .replace("{target_content}", target.getContent());
        messages.add(new UserMessage(request));

        String messageString = getRepresentation(messages);

        String cachedResponse = cache.get(messageString, String.class);
        if (cachedResponse != null && responseProblem(cachedResponse) == null) {
            return cachedResponse;
        } else {
            logger.info(
                    "Classifying ({}): {} and {}",
                    provider.modelName(),
                    source.getIdentifier(),
                    target.getIdentifier());
            List<ChatMessage> requestMessages = messages;
            String lastProblem = "no response was received";
            for (int attempt = 1; attempt <= MAX_RESPONSE_ATTEMPTS; attempt++) {
                ChatResponse response = llm.chat(requestMessages);
                String responseText = responseText(response);
                lastProblem = responseProblem(responseText);
                if (response.finishReason() == dev.langchain4j.model.output.FinishReason.LENGTH) {
                    lastProblem = "response ended at the output token limit before a reliable final decision";
                }
                if (lastProblem == null) {
                    cache.put(messageString, responseText);
                    return responseText;
                }

                logger.warn(
                        "Invalid LLM classification response on attempt {}/{} for {} and {}: {} (finish reason: {})",
                        attempt,
                        MAX_RESPONSE_ATTEMPTS,
                        source.getIdentifier(),
                        target.getIdentifier(),
                        lastProblem,
                        response.finishReason());
                requestMessages = repairMessages(messages, lastProblem);
            }
            throw new IllegalStateException("LLM returned no parseable classification after %d attempts for %s and %s: %s"
                    .formatted(
                            MAX_RESPONSE_ATTEMPTS,
                            source.getIdentifier(),
                            target.getIdentifier(),
                            lastProblem));
        }
    }

    private @Nullable String responseText(ChatResponse response) {
        String text = response.aiMessage().text();
        String thinking = response.aiMessage().thinking();
        if (text == null || text.isBlank()) {
            return thinking;
        }
        return text;
    }

    private @Nullable String responseProblem(@Nullable String responseText) {
        if (responseText == null || responseText.isBlank()) {
            return "both content and thinking are empty";
        }

        if (classifyRelationType) {
            try {
                parseTraceDecision(responseText);
                return null;
            } catch (IllegalArgumentException e) {
                return e.getMessage();
            }
        }
        if (!containsTraceTag(responseText)) {
            return "the <trace> tag is missing";
        }
        return null;
    }

    private static boolean containsTraceTag(String responseText) {
        return Pattern.compile("<trace>\\s*(?:yes|no)\\s*</trace>", Pattern.CASE_INSENSITIVE | Pattern.DOTALL)
                .matcher(responseText)
                .find();
    }

    private static List<ChatMessage> repairMessages(List<ChatMessage> originalMessages, String problem) {
        List<ChatMessage> repaired = new ArrayList<>(originalMessages);
        repaired.add(new UserMessage("Re-evaluate the two artifacts above (reason: " + problem
                + "). Keep reasoning short and finish with exactly one complete pair: "
                + "<trace>no</trace><relation_type>None</relation_type>, "
                + "<trace>yes</trace><relation_type>Refine</relation_type>, or "
                + "<trace>yes</trace><relation_type>Satisfy</relation_type>."));
        return repaired;
    }

    private String getRepresentation(List<ChatMessage> messages) {
        List<String> messageStrings =
                messages.stream().map(this::getRepresentation).toList();
        return messageStrings.toString();
    }

    private String getRepresentation(ChatMessage message) {
        return switch (message) {
            case SystemMessage systemMessage -> "SystemMessage { text = %s }".formatted(quoted(systemMessage.text()));
            case UserMessage userMessage -> {
                List<String> content = userMessage.contents().stream()
                        .map(this::getRepresentation)
                        .toList();
                yield "UserMessage { name = %s contents = %s }".formatted(quoted(userMessage.name()), content);
            }

            default ->
                throw new IllegalStateException("Unexpected message type: %s. Expected SystemMessage or UserMessage"
                        .formatted(message.getClass().getName()));
        };
    }

    private String getRepresentation(Content content) {
        return switch (content) {
            case TextContent textContent -> "TextContent { text = %s }".formatted(quoted(textContent.text()));
            default ->
                throw new IllegalStateException("Unexpected content type: %s. Expected TextContent"
                        .formatted(content.getClass().getName()));
        };
    }

    record TraceDecision(boolean related, String relationType) {}
}

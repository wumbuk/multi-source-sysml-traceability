/* Licensed under MIT 2025-2026. */
package edu.kit.kastel.sdq.lissa.ratlr.classifier;

import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Base64;
import java.util.Map;

import edu.kit.kastel.sdq.lissa.ratlr.cache.CacheParameter;
import edu.kit.kastel.sdq.lissa.ratlr.cache.classifier.ClassifierCacheParameter;
import edu.kit.kastel.sdq.lissa.ratlr.configuration.ModuleConfiguration;
import edu.kit.kastel.sdq.lissa.ratlr.utils.Environment;

import dev.langchain4j.http.client.jdk.JdkHttpClient;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.ollama.OllamaChatModel;
import dev.langchain4j.model.openai.OpenAiChatModel;

/**
 * Provides chat language model instances for different platforms.
 * This class supports multiple language model platforms (OpenAI, Ollama, Blablador)
 * and handles their configuration, including authentication and model settings.
 * <p>
 * Required environment variables for each platform:
 * <ul>
 *   <li>OpenAI:
 *     <ul>
 *       <li>{@code LLM_BASE_URL}: OpenAI-compatible API base URL</li>
 *       <li>{@code LLM_MODEL}: Served model name</li>
 *       <li>{@code LLM_API_KEY}: API key (may be a placeholder for local vLLM)</li>
 *       <li>{@code LLM_TIMEOUT_SECONDS}: Client timeout in seconds</li>
 *       <li>{@code LLM_MAX_CONCURRENCY}: Maximum number of concurrent classification requests</li>
 *       <li>{@code LLM_MAX_TOKENS}: Optional per-response token cap for OpenAI-compatible models</li>
 *       <li>{@code LLM_ENABLE_THINKING}: Optional Qwen/vLLM per-request thinking mode (true or false)</li>
 *     </ul>
 *   </li>
 *   <li>Ollama:
 *     <ul>
 *       <li>{@code OLLAMA_HOST}: The host URL for the Ollama server</li>
 *       <li>{@code OLLAMA_USER}: Username for Ollama authentication (optional)</li>
 *       <li>{@code OLLAMA_PASSWORD}: Password for Ollama authentication (optional)</li>
 *     </ul>
 *   </li>
 *   <li>Blablador:
 *     <ul>
 *       <li>{@code BLABLADOR_API_KEY}: Your Blablador API key</li>
 *     </ul>
 *   </li>
 *   <li>DeepSeek:
 *   <ul>
 *       <li>{@code DEEPSEEK_API_KEY}: Your DeepSeek API key</li>
 *     </ul>
 *   </li>
 *   <li>Open WebUI:
 *   <ul>
 *       <li>{@code OPENWEBUI_URL}: The URL for the Open WebUI server</li>
 *       <li>{@code OPENWEBUI_API_KEY}: Your Open WebUI API key</li>
 *     </ul>
 *   </li>
 * </ul>
 *
 * @see ChatLanguageModelPlatform
 */
public class ChatLanguageModelProvider {
    public static final String DEFAULT_LLM_BASE_URL = "http://127.0.0.1:8000/v1";
    public static final String DEFAULT_LLM_MODEL = "Qwen3.5-27B";
    public static final String DEFAULT_LLM_API_KEY = "EMPTY";
    public static final long DEFAULT_LLM_TIMEOUT_SECONDS = 86_400L;

    /**
     * Default seed value for model.
     */
    public static final int DEFAULT_SEED = 133742243;

    /**
     * Default temperature setting for the model.
     */
    public static final double DEFAULT_TEMPERATURE = 0.0;

    /**
     * The platform to use for the language model.
     */
    private final ChatLanguageModelPlatform platform;

    /**
     * The name of the model to use.
     */
    private String modelName;

    /**
     * The seed value for model randomization.
     */
    private int seed;

    /**
     * Temperature setting for the model.
     */
    private double temperature;

    /**
     * Creates a new chat language model provider with the specified configuration.
     * The configuration name should be in the format "mode_platform" (e.g., "simple_openai").
     *
     * @param configuration The module configuration containing model settings
     */
    public ChatLanguageModelProvider(ModuleConfiguration configuration) {
        this.platform = ChatLanguageModelPlatform.fromModuleConfiguration(configuration);
        this.initPlatformParameters(configuration);
    }

    /**
     * Creates a chat model instance based on the configured platform.
     *
     * @return A chat model instance for the configured platform
     * @throws IllegalArgumentException If the platform is not supported
     */
    public ChatModel createChatModel() {
        return switch (platform) {
            case OPENAI -> createOpenAiChatModel(modelName, seed, temperature);
            case OLLAMA -> createOllamaChatModel(modelName, seed, temperature);
            case BLABLADOR -> createBlabladorChatModel(modelName, seed, temperature);
            case DEEPSEEK -> createDeepSeekChatModel(modelName, seed, temperature);
            case OPENWEBUI -> createOpenWebUIChatModel(modelName, seed, temperature);
        };
    }

    /**
     * Initializes the model platform settings from the configuration.
     * Sets the model name and seed value based on the platform.
     *
     * @param configuration The module configuration containing model settings
     * @throws IllegalArgumentException If the platform is not supported
     */
    private void initPlatformParameters(ModuleConfiguration configuration) {
        final String modelKey = "model";
        String configuredModel = configuration.argumentAsString(modelKey, platform.getDefaultModel());
        this.modelName = platform == ChatLanguageModelPlatform.OPENAI
                ? environmentOrDefault("LLM_MODEL", DEFAULT_LLM_MODEL)
                : configuredModel;
        this.seed = configuration.argumentAsInt("seed", DEFAULT_SEED);
        this.temperature = configuration.argumentAsDouble("temperature", DEFAULT_TEMPERATURE);
    }

    /**
     * Gets the name of the configured model.
     *
     * @return The model name
     */
    public String modelName() {
        return modelName;
    }

    /**
     * Gets the seed value used for model randomization.
     *
     * @return The seed value
     */
    public int seed() {
        return seed;
    }

    /**
     * Gets the temperature setting for the model.
     *
     * @return The temperature value
     */
    public double temperature() {
        return temperature;
    }

    /**
     * Determines the number of threads to use based on the platform.
     * OpenAI and Blablador platforms use 100 threads, while others use 1.
     *
     * @param configuration The module configuration
     * @return The number of threads to use
     */
    public static int threads(ModuleConfiguration configuration) {
        ChatLanguageModelPlatform platform = ChatLanguageModelPlatform.fromModuleConfiguration(configuration);
        if (platform != ChatLanguageModelPlatform.OPENAI) {
            return platform.getThreads();
        }

        long configured = positiveLongEnvironmentOrDefault("LLM_MAX_CONCURRENCY", platform.getThreads());
        if (configured > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("LLM_MAX_CONCURRENCY must not exceed " + Integer.MAX_VALUE);
        }
        return (int) configured;
    }

    /**
     * Creates an Ollama chat model instance.
     * The model is configured with authentication if credentials are provided.
     *
     * @param model The name of the model to use
     * @param seed The seed value for randomization
     * @param temperature The temperature setting for the model
     * @return A configured Ollama chat model instance
     */
    private static ChatModel createOllamaChatModel(String model, int seed, double temperature) {
        String host = Environment.getenv("OLLAMA_HOST");
        String user = Environment.getenv("OLLAMA_USER");
        String password = Environment.getenv("OLLAMA_PASSWORD");

        if (host == null) {
            throw new IllegalStateException("OLLAMA_HOST environment variable not set");
        }

        return new LazyChatModel(() -> {
            var ollama = OllamaChatModel.builder()
                    .baseUrl(host)
                    .modelName(model)
                    .timeout(Duration.ofMinutes(10))
                    .temperature(temperature)
                    .seed(seed);
            if (user != null && password != null && !user.isEmpty() && !password.isEmpty()) {
                ollama.customHeaders(Map.of(
                        "Authorization",
                        "Basic "
                                + Base64.getEncoder()
                                        .encodeToString((user + ":" + password).getBytes(StandardCharsets.UTF_8))));
            }
            return ollama.build();
        });
    }

    /**
     * Creates an OpenAI-compatible chat model instance. Connection settings come from
     * {@code LLM_BASE_URL}, {@code LLM_MODEL}, and {@code LLM_API_KEY}, with local-vLLM defaults.
     *
     * @param model The name of the model to use
     * @param seed The seed value for randomization
     * @param temperature The temperature setting for the model
     * @return A configured OpenAI chat model instance
     * @throws IllegalArgumentException If {@code LLM_TIMEOUT_SECONDS} is not a positive integer
     */
    private static ChatModel createOpenAiChatModel(String model, int seed, double temperature) {
        String baseUrl = environmentOrDefault("LLM_BASE_URL", DEFAULT_LLM_BASE_URL);
        String apiKey = environmentOrDefault("LLM_API_KEY", DEFAULT_LLM_API_KEY);
        long timeoutSeconds = positiveLongEnvironmentOrDefault("LLM_TIMEOUT_SECONDS", DEFAULT_LLM_TIMEOUT_SECONDS);
        String maxTokensSetting = Environment.getenv("LLM_MAX_TOKENS");
        Integer maxTokens = null;
        if (maxTokensSetting != null && !maxTokensSetting.isBlank()) {
            long parsed = positiveLongEnvironmentOrDefault("LLM_MAX_TOKENS", 0);
            if (parsed > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("LLM_MAX_TOKENS must not exceed " + Integer.MAX_VALUE);
            }
            maxTokens = (int) parsed;
        }
        final Integer responseTokenLimit = maxTokens;
        String thinkingSetting = Environment.getenv("LLM_ENABLE_THINKING");
        Boolean enableThinking = null;
        if (thinkingSetting != null && !thinkingSetting.isBlank()) {
            String normalized = thinkingSetting.trim();
            if (!normalized.equalsIgnoreCase("true") && !normalized.equalsIgnoreCase("false")) {
                throw new IllegalArgumentException("LLM_ENABLE_THINKING must be true or false");
            }
            enableThinking = Boolean.parseBoolean(normalized);
        }
        final Boolean requestThinkingMode = enableThinking;
        var httpClientBuilder = java.net.http.HttpClient.newBuilder()
                .version(java.net.http.HttpClient.Version.HTTP_1_1);

        return new LazyChatModel(() -> {
            var builder = new OpenAiChatModel.OpenAiChatModelBuilder()
                    .httpClientBuilder(JdkHttpClient.builder().httpClientBuilder(httpClientBuilder))
                    .baseUrl(baseUrl)
                    .modelName(model)
                    .apiKey(apiKey)
                    .temperature(temperature)
                    .seed(seed)
                    .timeout(Duration.ofSeconds(timeoutSeconds));
            if (responseTokenLimit != null) {
                builder.maxTokens(responseTokenLimit);
            }
            if (requestThinkingMode != null) {
                builder.customParameters(Map.of("chat_template_kwargs", Map.of("enable_thinking", requestThinkingMode)));
            }
            return builder.build();
        });
    }

    private static String environmentOrDefault(String name, String defaultValue) {
        String value = Environment.getenv(name);
        return value == null || value.isBlank() ? defaultValue : value.trim();
    }

    private static long positiveLongEnvironmentOrDefault(String name, long defaultValue) {
        String value = Environment.getenv(name);
        if (value == null || value.isBlank()) {
            return defaultValue;
        }
        try {
            long parsed = Long.parseLong(value.trim());
            if (parsed <= 0) {
                throw new NumberFormatException("value is not positive");
            }
            return parsed;
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(name + " must be a positive integer, but was '" + value + "'", e);
        }
    }

    /**
     * Creates a Blablador chat model instance.
     * Requires Blablador API key to be set in environment variables.
     *
     * @param model The name of the model to use
     * @param seed The seed value for randomization
     * @param temperature The temperature setting for the model
     * @return A configured Blablador chat model instance
     * @throws IllegalStateException If required environment variables are not set
     */
    private static ChatModel createBlabladorChatModel(String model, int seed, double temperature) {
        String blabladorApiKey = Environment.getenv("BLABLADOR_API_KEY");
        if (blabladorApiKey == null) {
            throw new IllegalStateException("BLABLADOR_API_KEY environment variable not set");
        }
        return new LazyChatModel(() -> new OpenAiChatModel.OpenAiChatModelBuilder()
                .baseUrl("https://api.helmholtz-blablador.fz-juelich.de/v1")
                .modelName(model)
                .apiKey(blabladorApiKey)
                .temperature(temperature)
                .seed(seed)
                .build());
    }

    /**
     * Creates a DeepSeek chat model instance.
     * Requires DeepSeek API key to be set in environment variables.
     *
     * @param model The name of the model to use
     * @param seed The seed value for randomization
     * @param temperature The temperature setting for the model
     * @return A configured DeepSeek chat model instance
     * @throws IllegalStateException If required environment variables are not set
     */
    private static ChatModel createDeepSeekChatModel(String model, int seed, double temperature) {
        String deepseekApiKey = Environment.getenv("DEEPSEEK_API_KEY");
        if (deepseekApiKey == null) {
            throw new IllegalStateException("DEEPSEEK_API_KEY environment variable not set");
        }
        return new LazyChatModel(() -> new OpenAiChatModel.OpenAiChatModelBuilder()
                .baseUrl("https://api.deepseek.com/v1")
                .modelName(model)
                .apiKey(deepseekApiKey)
                .temperature(temperature)
                .seed(seed)
                .build());
    }

    /**
     * Creates an Open WebUI chat model instance.
     * Requires Open WebUI API key and url to be set in environment variables.
     *
     * @param model The name of the model to use
     * @param seed The seed value for randomization
     * @param temperature The temperature setting for the model
     * @return A configured Open WebUI chat model instance
     * @throws IllegalStateException If required environment variables are not set
     */
    private static ChatModel createOpenWebUIChatModel(String model, int seed, double temperature) {
        String openwebuiUrl = Environment.getenv("OPENWEBUI_URL");
        String openwebuiApiKey = Environment.getenv("OPENWEBUI_API_KEY");
        if (openwebuiUrl == null || openwebuiApiKey == null) {
            throw new IllegalStateException("OPENWEBUI_URL or OPENWEBUI_API_KEY environment variable not set");
        }

        return new LazyChatModel(() -> new OpenAiChatModel.OpenAiChatModelBuilder()
                .baseUrl(openwebuiUrl)
                .modelName(model)
                .apiKey(openwebuiApiKey)
                .temperature(temperature)
                .seed(seed)
                .timeout(Duration.ofMinutes(10))
                .build());
    }

    /**
     * Returns the parameters used to create the cache key for this model.
     * This method is used to identify the cache uniquely.
     *
     * @return An array of strings representing the cache parameters
     * @see edu.kit.kastel.sdq.lissa.ratlr.cache.CacheManager#getCache(Object, CacheParameter)
     */
    public ClassifierCacheParameter cacheParameters() {
        return new ClassifierCacheParameter(modelName, seed, temperature);
    }
}

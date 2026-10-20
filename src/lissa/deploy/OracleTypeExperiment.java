import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.output.FinishReason;
import edu.kit.kastel.sdq.lissa.ratlr.classifier.ChatLanguageModelProvider;
import edu.kit.kastel.sdq.lissa.ratlr.configuration.ModuleConfiguration;
import edu.kit.kastel.sdq.lissa.ratlr.context.ContextStore;
import edu.kit.kastel.sdq.lissa.ratlr.dataset.PairCsvDataset;
import edu.kit.kastel.sdq.lissa.ratlr.knowledge.Artifact;
import edu.kit.kastel.sdq.lissa.ratlr.knowledge.Element;
import edu.kit.kastel.sdq.lissa.ratlr.preprocessor.SingleArtifactPreprocessor;

// Run with Java 21 source launcher and the built LiSSA JAR. No classification cache.
class OracleTypeExperiment {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final Pattern FINAL_TYPE = Pattern.compile(
            "<relation_type>\\s*(Refine|Satisfy)\\s*</relation_type>\\s*\\z",
            Pattern.CASE_INSENSITIVE);

    private static Map<String, Element> elements(PairCsvDataset dataset, JsonNode config, String key) {
        var args = config.path(key).path("args");
        var artifacts = dataset.artifacts(PairCsvDataset.Side.from(args.path("side").asText()),
                Artifact.ArtifactType.from(args.path("artifact_type").asText()),
                PairCsvDataset.ContentConflictStrategy.from(args.path("content_conflict").asText("merge")));
        var result = new LinkedHashMap<String, Element>();
        new SingleArtifactPreprocessor(new ContextStore()).preprocess(artifacts)
                .forEach(element -> result.put(element.getIdentifier(), element));
        return result;
    }

    private static void append(Path path, Object value) throws Exception {
        Files.writeString(path, JSON.writeValueAsString(value) + "\n", StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.APPEND);
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 4) {
            throw new IllegalArgumentException("Usage: OracleTypeExperiment <suite> <dataset> <round> <output>");
        }
        var suite = Path.of(args[0]);
        String datasetName = args[1];
        int round = Integer.parseInt(args[2]);
        if (!List.of("Air_Condition", "Extract_Requirement_Values", "Architecting_Spacecraft").contains(datasetName)
                || round < 1 || round > 3) {
            throw new IllegalArgumentException("Unsupported dataset or round");
        }
        if (!"true".equalsIgnoreCase(System.getenv("LLM_ENABLE_THINKING"))) {
            throw new IllegalStateException("Oracle run requires LLM_ENABLE_THINKING=true");
        }
        String cap = System.getenv("LLM_MAX_TOKENS");
        if (cap != null && !cap.isBlank()) {
            throw new IllegalStateException("Oracle run must not set LLM_MAX_TOKENS");
        }
        var output = Path.of(args[3]);
        Files.createDirectories(output);
        for (String name : List.of("predictions.json", "predictions.jsonl", "responses.jsonl", "run_info.json")) {
            if (Files.exists(output.resolve(name))) {
                throw new IllegalStateException("Refusing to overwrite " + output.resolve(name));
            }
        }
        var input = suite.resolve("_inputs").resolve(datasetName);
        var config = JSON.readTree(input.resolve("config.json").toFile());
        for (String key : List.of("source_preprocessor", "target_preprocessor")) {
            if (!config.path(key).path("name").asText().equals("artifact")) {
                throw new IllegalArgumentException("Oracle adapter requires the current artifact preprocessing");
            }
        }
        // Build text from the complete CSV exactly as in the original typed experiment.
        // PairCsvDataset's artifact rendering excludes the Relation column.
        var dataset = PairCsvDataset.load(input.resolve("Judge.csv"));
        var sources = elements(dataset, config, "source_artifact_provider");
        var targets = elements(dataset, config, "target_artifact_provider");
        // cases.json contains IDs only. Gold relation types are held by the evaluator.
        var cases = JSON.readTree(input.resolve("cases.json").toFile());
        var prompt = JSON.readTree(suite.resolve("_inputs/oracle_type_prompt.json").toFile());
        var settings = new LinkedHashMap<String, String>();
        config.path("classifier").path("args").fields()
                .forEachRemaining(entry -> settings.put(entry.getKey(), entry.getValue().asText()));
        var provider = new ChatLanguageModelProvider(new ModuleConfiguration("reasoning_openai", settings));
        var model = provider.createChatModel();
        var info = new LinkedHashMap<String, Object>();
        info.put("protocol", "oracle_known_existence_type_only");
        info.put("dataset", datasetName);
        info.put("round", round);
        info.put("expected_predictions", cases.size());
        info.put("model", provider.modelName());
        info.put("temperature", provider.temperature());
        info.put("seed", provider.seed());
        info.put("thinking", true);
        info.put("client_max_tokens", null);
        info.put("cache_enabled", false);
        info.put("started_at", Instant.now().toString());
        JSON.writerWithDefaultPrettyPrinter().writeValue(output.resolve("run_info.json").toFile(), info);

        var predictions = new ArrayList<Map<String, Object>>();
        for (var item : cases) {
            String sourceId = item.path("source_id").asText();
            String targetId = item.path("target_id").asText();
            var source = sources.get(sourceId);
            var target = targets.get(targetId);
            if (source == null || target == null) {
                throw new IllegalStateException("Oracle pair missing artifact: " + sourceId + " -> " + targetId);
            }
            String request = prompt.path("user").asText()
                    .replace("{source_type}", source.getType()).replace("{source_content}", source.getContent())
                    .replace("{target_type}", target.getType()).replace("{target_content}", target.getContent());
            String system = prompt.path("system").asText();
            String prediction = null;
            int acceptedAttempt = 0;
            String problem = "";
            for (int attempt = 1; attempt <= 3; attempt++) {
                List<ChatMessage> messages = new ArrayList<>();
                messages.add(new SystemMessage(system));
                messages.add(new UserMessage(request));
                String repair = attempt == 1 ? "" : "Re-evaluate the type of the known link above. "
                        + "The previous response was invalid: " + problem
                        + ". Keep reasoning concise and end with exactly one complete tag: "
                        + "<relation_type>Refine</relation_type> or <relation_type>Satisfy</relation_type>.";
                if (!repair.isEmpty()) messages.add(new UserMessage(repair));
                var raw = new LinkedHashMap<String, Object>();
                raw.put("source_id", sourceId);
                raw.put("target_id", targetId);
                raw.put("attempt", attempt);
                raw.put("system", system);
                raw.put("prompt", request);
                raw.put("repair", repair);
                var start = Instant.now();
                try {
                    var response = model.chat(messages);
                    String text = response.aiMessage().text();
                    String thinking = response.aiMessage().thinking();
                    String parseText = text == null || text.isBlank() ? thinking : text;
                    var match = FINAL_TYPE.matcher(parseText == null ? "" : parseText);
                    boolean valid = response.finishReason() == FinishReason.STOP && match.find();
                    raw.put("content", text);
                    raw.put("thinking", thinking);
                    raw.put("finish_reason", String.valueOf(response.finishReason()));
                    raw.put("token_usage", String.valueOf(response.tokenUsage()));
                    raw.put("valid", valid);
                    raw.put("elapsed_ms", java.time.Duration.between(start, Instant.now()).toMillis());
                    append(output.resolve("responses.jsonl"), raw);
                    if (valid) {
                        prediction = match.group(1).equalsIgnoreCase("Refine") ? "Refine" : "Satisfy";
                        acceptedAttempt = attempt;
                        break;
                    }
                    problem = "finish_reason=" + response.finishReason() + " or missing final valid type tag";
                    System.err.printf("Invalid Oracle response: %d/%d, attempt %d/3, %s%n",
                            predictions.size() + 1, cases.size(), attempt, problem);
                } catch (RuntimeException error) {
                    raw.put("error", error.toString());
                    append(output.resolve("responses.jsonl"), raw);
                    throw error;
                }
            }
            if (prediction == null) throw new IllegalStateException("No valid Oracle type after 3 attempts: " + sourceId);
            var record = new LinkedHashMap<String, Object>();
            record.put("dataset", datasetName);
            record.put("round", round);
            record.put("source_id", sourceId);
            record.put("target_id", targetId);
            record.put("predicted_type", prediction);
            record.put("accepted_attempt", acceptedAttempt);
            predictions.add(record);
            append(output.resolve("predictions.jsonl"), record);
            System.out.printf("ORACLE_TYPE_DECISION\t%d/%d\t%s\t%s\t%s%n",
                    predictions.size(), cases.size(), sourceId, targetId, prediction);
        }
        JSON.writerWithDefaultPrettyPrinter().writeValue(output.resolve("predictions.json").toFile(), predictions);
        System.out.printf("Oracle %s round_%d completed: %d fresh predictions%n", datasetName, round, predictions.size());
    }
}

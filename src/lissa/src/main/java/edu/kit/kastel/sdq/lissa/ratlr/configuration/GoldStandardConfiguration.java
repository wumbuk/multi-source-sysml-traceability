/* Licensed under MIT 2025-2026. */
package edu.kit.kastel.sdq.lissa.ratlr.configuration;

import java.io.IOException;
import java.nio.file.Path;

import org.jspecify.annotations.Nullable;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Represents the configuration for gold standard evaluation in trace link analysis.
 * This record contains settings for loading and parsing gold standard data from a file.
 * The gold standard file is expected to be a CSV file containing trace links between
 * source and target elements.
 *
 * @param path Path to the gold standard file. The file should be a CSV file containing trace links.
 * @param hasHeader Whether the gold standard file has a header row.
 *                  If true, the first row will be skipped during parsing.
 * @param swapColumns Whether to swap the source and target columns when reading the file.
 *                    This is useful when the gold standard file has columns in a different order
 *                    than expected by the system.
 * @param format Either {@code trace_links} for a two-column file or {@code pair_csv} for a labeled pair dataset.
 * @param sourceSide Source side used when {@code format=pair_csv}.
 * @param targetSide Target side used when {@code format=pair_csv}.
 * @param relationColumn Relation-label column used when {@code format=pair_csv}.
 * @param positiveRelationPrefix Prefix identifying positive labels when {@code format=pair_csv}.
 */
public record GoldStandardConfiguration(
        @JsonProperty("path") String path,
        @JsonProperty(defaultValue = "false") boolean hasHeader,

        @JsonProperty(value = "swap_columns", defaultValue = "false")
        boolean swapColumns,

        @JsonProperty(value = "format", defaultValue = "trace_links")
        String format,

        @JsonProperty(value = "source_side", defaultValue = "B")
        String sourceSide,

        @JsonProperty(value = "target_side", defaultValue = "A")
        String targetSide,

        @JsonProperty(value = "relation_column", defaultValue = "Relation")
        String relationColumn,

        @JsonProperty(value = "positive_relation_prefix", defaultValue = "1")
        String positiveRelationPrefix)
        implements Configuration {

    public GoldStandardConfiguration {
        format = defaultIfBlank(format, "trace_links");
        sourceSide = defaultIfBlank(sourceSide, "B");
        targetSide = defaultIfBlank(targetSide, "A");
        relationColumn = defaultIfBlank(relationColumn, "Relation");
        positiveRelationPrefix = defaultIfBlank(positiveRelationPrefix, "1");
    }

    /**
     * Loads a gold standard configuration from a JSON file.
     * If the file cannot be loaded or is null, returns null.
     *
     * @param evaluationConfig Path to the JSON configuration file
     * @return A new GoldStandardConfiguration instance, or null if loading fails
     */
    public static @Nullable GoldStandardConfiguration load(@Nullable Path evaluationConfig) {
        if (evaluationConfig == null) return null;

        try {
            return new ObjectMapper().readValue(evaluationConfig.toFile(), GoldStandardConfiguration.class);
        } catch (IOException e) {
            LoggerFactory.getLogger(GoldStandardConfiguration.class)
                    .error("Loading evaluation config threw an exception: {}", e.getMessage());
            return null;
        }
    }

    private static String defaultIfBlank(@Nullable String value, String defaultValue) {
        return value == null || value.isBlank() ? defaultValue : value;
    }
}

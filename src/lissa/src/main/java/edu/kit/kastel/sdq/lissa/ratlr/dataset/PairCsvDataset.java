/* Licensed under MIT 2025-2026. */
package edu.kit.kastel.sdq.lissa.ratlr.dataset;

import java.io.IOException;
import java.io.PushbackReader;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import edu.kit.kastel.sdq.lissa.ratlr.knowledge.Artifact;
import edu.kit.kastel.sdq.lissa.ratlr.knowledge.TraceLink;

/**
 * Loads the paired CSV format used by the trace-relation datasets.
 * Each row describes one candidate pair with columns for side A, side B, and a relation label.
 */
public final class PairCsvDataset {
    private static final List<String> REQUIRED_COLUMNS = List.of(
            "NameA",
            "pathA",
            "contextA",
            "propertyA",
            "typeA",
            "NameB",
            "pathB",
            "contextB",
            "propertyB",
            "typeB",
            "Relation");

    private final Path path;
    private final List<Map<String, String>> rows;

    private PairCsvDataset(Path path, List<Map<String, String>> rows) {
        this.path = path;
        this.rows = rows;
    }

    /** Loads and validates a paired CSV dataset as UTF-8. */
    public static PairCsvDataset load(Path path) {
        if (!Files.isRegularFile(path)) {
            throw new IllegalArgumentException(
                    "Pair CSV dataset does not exist or is not a file: " + path.toAbsolutePath());
        }
        List<List<String>> records = readCsv(path);
        if (records.isEmpty()) {
            throw new IllegalArgumentException("Pair CSV dataset is empty: " + path.toAbsolutePath());
        }

        List<String> header = new ArrayList<>(records.getFirst());
        if (!header.isEmpty()) {
            header.set(0, removeBom(header.getFirst()));
        }
        if (new LinkedHashSet<>(header).size() != header.size()) {
            throw new IllegalArgumentException("Pair CSV dataset has duplicate column names: " + path);
        }
        List<String> missing = REQUIRED_COLUMNS.stream()
                .filter(column -> !header.contains(column))
                .toList();
        if (!missing.isEmpty()) {
            throw new IllegalArgumentException("Pair CSV dataset is missing required columns " + missing + ": " + path);
        }

        List<Map<String, String>> rows = new ArrayList<>();
        for (int index = 1; index < records.size(); index++) {
            List<String> record = records.get(index);
            if (record.size() != header.size()) {
                throw new IllegalArgumentException("Pair CSV row " + (index + 1) + " has " + record.size()
                        + " fields, expected " + header.size() + ": " + path);
            }
            Map<String, String> row = new LinkedHashMap<>();
            for (int column = 0; column < header.size(); column++) {
                row.put(header.get(column), record.get(column));
            }
            rows.add(Collections.unmodifiableMap(row));
        }
        if (rows.isEmpty()) {
            throw new IllegalArgumentException("Pair CSV dataset has a header but no data rows: " + path);
        }
        return new PairCsvDataset(path, List.copyOf(rows));
    }

    /** Creates one deduplicated artifact per unique entity on the selected side. */
    public List<Artifact> artifacts(
            Side side, Artifact.ArtifactType artifactType, ContentConflictStrategy conflictStrategy) {
        Map<String, List<Map<String, String>>> grouped = new LinkedHashMap<>();
        for (Map<String, String> row : rows) {
            String identifier = identifier(side, row);
            grouped.computeIfAbsent(identifier, ignored -> new ArrayList<>()).add(row);
        }

        List<Artifact> artifacts = new ArrayList<>();
        for (var entry : grouped.entrySet()) {
            artifacts.add(new Artifact(
                    entry.getKey(), artifactType, renderArtifact(side, entry.getValue(), conflictStrategy)));
        }
        artifacts.sort(Comparator.comparing(Artifact::getIdentifier));
        return List.copyOf(artifacts);
    }

    /** Returns all labeled candidate pairs, grouped by source identifier. */
    public Map<String, Set<String>> candidateTargets(Side sourceSide, Side targetSide) {
        ensureDifferentSides(sourceSide, targetSide);
        Map<String, Set<String>> candidates = new LinkedHashMap<>();
        for (Map<String, String> row : rows) {
            candidates
                    .computeIfAbsent(identifier(sourceSide, row), ignored -> new LinkedHashSet<>())
                    .add(identifier(targetSide, row));
        }
        Map<String, Set<String>> immutable = new LinkedHashMap<>();
        for (var entry : candidates.entrySet()) {
            immutable.put(entry.getKey(), Set.copyOf(entry.getValue()));
        }
        return Collections.unmodifiableMap(immutable);
    }

    /** Extracts positive trace links from the relation column. */
    public Set<TraceLink> positiveTraceLinks(
            Side sourceSide, Side targetSide, String relationColumn, String positiveRelationPrefix) {
        ensureDifferentSides(sourceSide, targetSide);
        if (positiveRelationPrefix == null || positiveRelationPrefix.isEmpty()) {
            throw new IllegalArgumentException("positive_relation_prefix must not be empty");
        }
        if (!rows.getFirst().containsKey(relationColumn)) {
            throw new IllegalArgumentException("Relation column '" + relationColumn + "' does not exist in " + path);
        }

        Set<TraceLink> result = new LinkedHashSet<>();
        for (Map<String, String> row : rows) {
            if (row.get(relationColumn).trim().startsWith(positiveRelationPrefix)) {
                result.add(new TraceLink(identifier(sourceSide, row), identifier(targetSide, row)));
            }
        }
        return Collections.unmodifiableSet(result);
    }

    /** Number of candidate-pair rows in the dataset. */
    public int size() {
        return rows.size();
    }

    private static String renderArtifact(
            Side side, List<Map<String, String>> entityRows, ContentConflictStrategy conflictStrategy) {
        String suffix = side.name();
        StringBuilder result = new StringBuilder();
        appendValues(result, "Name", values(entityRows, "Name" + suffix), ContentConflictStrategy.ERROR);
        appendValues(result, "Path", values(entityRows, "path" + suffix), ContentConflictStrategy.ERROR);
        appendValues(result, "Type", values(entityRows, "type" + suffix), ContentConflictStrategy.ERROR);
        appendValues(result, "Property", values(entityRows, "property" + suffix), conflictStrategy);
        appendValues(result, "Context", values(entityRows, "context" + suffix), conflictStrategy);
        return result.toString().stripTrailing();
    }

    private static void appendValues(
            StringBuilder result, String label, List<String> values, ContentConflictStrategy conflictStrategy) {
        List<String> nonBlank = values.stream()
                .map(String::trim)
                .filter(value -> !value.isEmpty())
                .distinct()
                .toList();
        if (nonBlank.isEmpty()) {
            return;
        }
        if (nonBlank.size() > 1 && conflictStrategy == ContentConflictStrategy.ERROR) {
            throw new IllegalArgumentException(
                    "Conflicting " + label.toLowerCase(Locale.ROOT) + " values for one paired-CSV entity: " + nonBlank);
        }
        int limit = conflictStrategy == ContentConflictStrategy.FIRST ? 1 : nonBlank.size();
        for (int index = 0; index < limit; index++) {
            result.append(label);
            if (limit > 1) {
                result.append(' ').append(index + 1);
            }
            result.append(": ").append(nonBlank.get(index)).append('\n');
        }
    }

    private static List<String> values(List<Map<String, String>> rows, String column) {
        List<String> result = new ArrayList<>();
        for (Map<String, String> row : rows) {
            result.add(row.get(column));
        }
        return result;
    }

    private static String identifier(Side side, Map<String, String> row) {
        String suffix = side.name();
        return side.name() + "|" + encodeIdentifierPart(row.get("type" + suffix)) + "|"
                + encodeIdentifierPart(row.get("path" + suffix)) + "|"
                + encodeIdentifierPart(row.get("Name" + suffix));
    }

    private static String encodeIdentifierPart(String value) {
        return value.trim()
                .replace("%", "%25")
                .replace("|", "%7C")
                .replace(",", "%2C")
                .replace("\r", "%0D")
                .replace("\n", "%0A");
    }

    private static void ensureDifferentSides(Side sourceSide, Side targetSide) {
        if (sourceSide == targetSide) {
            throw new IllegalArgumentException("source_side and target_side must be different");
        }
    }

    private static String removeBom(String value) {
        return value.startsWith("\uFEFF") ? value.substring(1) : value;
    }

    private static List<List<String>> readCsv(Path path) {
        List<List<String>> records = new ArrayList<>();
        try (PushbackReader reader = new PushbackReader(Files.newBufferedReader(path, StandardCharsets.UTF_8), 1)) {
            List<String> record = new ArrayList<>();
            StringBuilder field = new StringBuilder();
            boolean quoted = false;
            int next;
            while ((next = reader.read()) != -1) {
                char current = (char) next;
                if (quoted) {
                    if (current == '"') {
                        int following = reader.read();
                        if (following == '"') {
                            field.append('"');
                        } else {
                            quoted = false;
                            if (following != -1) {
                                reader.unread(following);
                            }
                        }
                    } else {
                        field.append(current);
                    }
                } else if (current == '"' && field.isEmpty()) {
                    quoted = true;
                } else if (current == ',') {
                    record.add(field.toString());
                    field.setLength(0);
                } else if (current == '\n' || current == '\r') {
                    if (current == '\r') {
                        int following = reader.read();
                        if (following != '\n' && following != -1) {
                            reader.unread(following);
                        }
                    }
                    record.add(field.toString());
                    field.setLength(0);
                    if (!isBlankRecord(record)) {
                        records.add(List.copyOf(record));
                    }
                    record = new ArrayList<>();
                } else {
                    field.append(current);
                }
            }
            if (quoted) {
                throw new IllegalArgumentException("Unclosed quoted field in pair CSV dataset: " + path);
            }
            if (!record.isEmpty() || !field.isEmpty()) {
                record.add(field.toString());
                if (!isBlankRecord(record)) {
                    records.add(List.copyOf(record));
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return records;
    }

    private static boolean isBlankRecord(List<String> record) {
        return record.size() == 1 && record.getFirst().isBlank();
    }

    /** The entity side in the paired CSV schema. */
    public enum Side {
        A,
        B;

        public static Side from(String value) {
            try {
                return valueOf(value.trim().toUpperCase(Locale.ROOT));
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException("CSV side must be A or B, but was '" + value + "'", e);
            }
        }
    }

    /** How repeated entity rows with different semantic content are handled. */
    public enum ContentConflictStrategy {
        MERGE,
        FIRST,
        ERROR;

        public static ContentConflictStrategy from(String value) {
            try {
                return valueOf(value.trim().toUpperCase(Locale.ROOT));
            } catch (IllegalArgumentException e) {
                throw new IllegalArgumentException(
                        "content_conflict must be merge, first, or error, but was '" + value + "'", e);
            }
        }
    }
}

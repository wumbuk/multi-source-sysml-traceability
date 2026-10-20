/* Licensed under MIT 2025-2026. */
package edu.kit.kastel.sdq.lissa.ratlr.preprocessor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import edu.kit.kastel.sdq.lissa.ratlr.configuration.ModuleConfiguration;
import edu.kit.kastel.sdq.lissa.ratlr.context.ContextStore;
import edu.kit.kastel.sdq.lissa.ratlr.knowledge.Artifact;
import edu.kit.kastel.sdq.lissa.ratlr.knowledge.Element;

/**
 * Extracts traceable elements from textual SysML v2/KerML models.
 *
 * <p>This is deliberately a lightweight structural parser, not a replacement for the
 * normative SysML v2 parser. It recognizes common definition and usage declarations,
 * retains their nesting, and serializes local context for embedding and LLM classification.
 * Strict mode rejects files in which declarations cannot be structurally delimited,
 * preventing silent whole-file fallbacks in experiments.
 */
public final class SysMLV2Preprocessor extends Preprocessor {
    private static final String DEFAULT_TYPES =
            "requirement,part,port,item,action,state,interface,connection,constraint,attribute,use case,view,viewpoint,analysis,calculation,enum";
    private static final Pattern DECLARATION = Pattern.compile("(?im)(?<![\\w:])(?:(abstract|variation)\\s+)?"
            + "(package|requirement|part|port|item|action|state|interface|connection|constraint|attribute|use\\s+case|viewpoint|view|analysis|calculation|enum)"
            + "\\s+(?:(def)\\s+)?(?:<\\s*('[^']+'|[^>]+)\\s*>\\s*)?('[^']+'|[A-Za-z_][A-Za-z0-9_]*)");

    private final Set<String> includedTypes;
    private final Set<String> excludedTypes;
    private final boolean includeBody;
    private final boolean strict;

    public SysMLV2Preprocessor(ModuleConfiguration configuration, ContextStore contextStore) {
        super(contextStore);
        this.includedTypes = parseTypes(configuration.argumentAsString("include_types", DEFAULT_TYPES));
        this.excludedTypes = parseTypes(configuration.argumentAsString("exclude_types", "package"));
        this.includeBody = configuration.argumentAsBoolean("include_body", true);
        this.strict = configuration.argumentAsBoolean("strict", true);
    }

    @Override
    public List<Element> preprocess(List<Artifact> artifacts) {
        List<Element> elements = new ArrayList<>();
        for (Artifact artifact : artifacts) {
            Element root = new Element(
                    artifact.getIdentifier(),
                    "sysml model",
                    "SysML v2 model file: " + artifact.getIdentifier(),
                    0,
                    null,
                    false);
            elements.add(root);

            List<Node> nodes = parse(artifact.getContent(), artifact.getIdentifier());
            Map<Node, Element> nodeElements = new LinkedHashMap<>();
            Map<String, Integer> identifierCounts = new HashMap<>();
            for (Node node : nodes) {
                Element parent = node.parent == null ? root : nodeElements.get(node.parent);
                int granularity = parent.getGranularity() + 1;
                String segment = node.kind + (node.definition ? "-def" : "") + ":" + sanitizeId(node.name);
                String baseIdentifier = parent.getIdentifier() + SEPARATOR + segment;
                int occurrence = identifierCounts.merge(baseIdentifier, 1, Integer::sum);
                String identifier = occurrence == 1 ? baseIdentifier : baseIdentifier + "#" + occurrence;
                boolean compare = includedTypes.contains(node.kind) && !excludedTypes.contains(node.kind);
                Element element = new Element(
                        identifier,
                        "sysml " + node.kind + (node.definition ? " definition" : " usage"),
                        render(node, artifact.getContent()),
                        granularity,
                        parent,
                        compare);
                elements.add(element);
                nodeElements.put(node, element);
            }
        }
        return elements;
    }

    private List<Node> parse(String source, String identifier) {
        String structural = maskCommentsAndDoubleQuotedStrings(source);
        Matcher matcher = DECLARATION.matcher(structural);
        List<Node> nodes = new ArrayList<>();
        while (matcher.find()) {
            String kind = normalizeKind(matcher.group(2));
            boolean definition = matcher.group(3) != null;
            String shortName = unquote(matcher.group(4));
            String name = unquote(matcher.group(5));
            int delimiter = findDelimiter(structural, matcher.end());
            if (delimiter < 0) {
                if (strict) {
                    throw parseError(identifier, "No '{' or ';' after declaration " + kind + " " + name);
                }
                continue;
            }
            int end;
            int bodyStart = -1;
            if (structural.charAt(delimiter) == '{') {
                bodyStart = delimiter;
                end = findMatchingBrace(structural, delimiter);
                if (end < 0) {
                    if (strict) {
                        throw parseError(identifier, "Unclosed body for declaration " + kind + " " + name);
                    }
                    end = source.length() - 1;
                }
            } else {
                end = delimiter;
            }
            nodes.add(new Node(kind, definition, shortName, name, matcher.start(), bodyStart, end));
        }
        if (nodes.isEmpty() && strict) {
            throw parseError(identifier, "No supported SysML v2 declarations found");
        }

        for (Node node : nodes) {
            node.parent = nodes.stream()
                    .filter(candidate -> candidate != node)
                    .filter(candidate -> candidate.bodyStart >= 0)
                    .filter(candidate -> candidate.bodyStart < node.start && node.start < candidate.end)
                    .max((left, right) -> Integer.compare(left.bodyStart, right.bodyStart))
                    .orElse(null);
            if (node.parent != null) {
                node.parent.children.add(node);
            }
        }
        return nodes;
    }

    private String render(Node node, String source) {
        List<String> qualifiedParts = new ArrayList<>();
        for (Node current = node; current != null; current = current.parent) {
            qualifiedParts.add(0, current.name);
        }
        StringBuilder result = new StringBuilder();
        result.append("Element Type: ").append(node.kind).append('\n');
        result.append("Name: ").append(node.name).append('\n');
        if (node.shortName != null && !node.shortName.isBlank()) {
            result.append("Short Name: ").append(node.shortName).append('\n');
        }
        result.append("Definition: ").append(node.definition).append('\n');
        result.append("Qualified Name: ").append(String.join("::", qualifiedParts));
        if (node.parent != null) {
            result.append('\n')
                    .append("Parent: ")
                    .append(node.parent.kind)
                    .append(' ')
                    .append(node.parent.name);
        }
        if (includeBody) {
            String localBody = localBody(node, source);
            if (!localBody.isBlank()) {
                result.append('\n').append("Local Declaration: ").append(localBody);
            }
        }
        return result.toString();
    }

    private static String localBody(Node node, String source) {
        char[] local = source.substring(node.start, Math.min(node.end + 1, source.length()))
                .toCharArray();
        // Child declarations are represented separately. Removing them avoids duplicating a whole subtree in each
        // embedding.
        for (Node child : node.children()) {
            int from = Math.max(0, child.start - node.start);
            int to = Math.min(local.length, child.end + 1 - node.start);
            Arrays.fill(local, from, to, ' ');
        }
        return new String(local).replaceAll("\\s+", " ").trim();
    }

    private static int findDelimiter(String source, int start) {
        for (int i = start; i < source.length(); i++) {
            char current = source.charAt(i);
            if (current == '{' || current == ';') return i;
            if (current == '}') return -1;
        }
        return -1;
    }

    private static int findMatchingBrace(String source, int open) {
        int depth = 0;
        for (int i = open; i < source.length(); i++) {
            if (source.charAt(i) == '{') depth++;
            if (source.charAt(i) == '}' && --depth == 0) return i;
        }
        return -1;
    }

    private static String maskCommentsAndDoubleQuotedStrings(String source) {
        char[] masked = source.toCharArray();
        boolean lineComment = false;
        boolean blockComment = false;
        boolean string = false;
        boolean escaped = false;
        for (int i = 0; i < masked.length; i++) {
            char current = masked[i];
            char next = i + 1 < masked.length ? masked[i + 1] : '\0';
            if (lineComment) {
                if (current == '\n') lineComment = false;
                else masked[i] = ' ';
                continue;
            }
            if (blockComment) {
                if (current == '*' && next == '/') {
                    masked[i] = masked[i + 1] = ' ';
                    i++;
                    blockComment = false;
                } else if (current != '\n') {
                    masked[i] = ' ';
                }
                continue;
            }
            if (string) {
                if (current == '"' && !escaped) string = false;
                if (current != '\n') masked[i] = ' ';
                escaped = current == '\\' && !escaped;
                if (current != '\\') escaped = false;
                continue;
            }
            if (current == '/' && next == '/') {
                masked[i] = masked[i + 1] = ' ';
                i++;
                lineComment = true;
            } else if (current == '/' && next == '*') {
                masked[i] = masked[i + 1] = ' ';
                i++;
                blockComment = true;
            } else if (current == '"') {
                masked[i] = ' ';
                string = true;
            }
        }
        return new String(masked);
    }

    private static Set<String> parseTypes(String value) {
        if (value.equalsIgnoreCase("all")) {
            Set<String> result = new HashSet<>(parseTypes(DEFAULT_TYPES));
            result.add("package");
            return result;
        }
        return Arrays.stream(value.split(","))
                .map(SysMLV2Preprocessor::normalizeKind)
                .filter(it -> !it.isBlank())
                .collect(Collectors.toCollection(HashSet::new));
    }

    private static String normalizeKind(String kind) {
        return kind == null ? "" : kind.trim().toLowerCase(Locale.ROOT).replaceAll("\\s+", " ");
    }

    private static String unquote(String value) {
        if (value == null) return null;
        String trimmed = value.trim();
        if (trimmed.length() >= 2 && trimmed.startsWith("'") && trimmed.endsWith("'")) {
            return trimmed.substring(1, trimmed.length() - 1);
        }
        return trimmed;
    }

    private static String sanitizeId(String value) {
        return value.replace('$', '_').replace('\n', ' ').replace('\r', ' ').trim();
    }

    private static IllegalArgumentException parseError(String identifier, String message) {
        return new IllegalArgumentException("Cannot parse SysML v2 artifact '" + identifier + "': " + message);
    }

    private static final class Node {
        private final String kind;
        private final boolean definition;
        private final String shortName;
        private final String name;
        private final int start;

        private final int bodyStart;
        private final int end;
        private final List<Node> children = new ArrayList<>();
        private Node parent;

        private Node(
                String kind, boolean definition, String shortName, String name, int start, int bodyStart, int end) {
            this.kind = kind;
            this.definition = definition;
            this.shortName = shortName;
            this.name = name;
            this.start = start;
            this.bodyStart = bodyStart;
            this.end = end;
        }

        private List<Node> children() {
            return children;
        }
    }
}

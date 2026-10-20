/* Licensed under MIT 2025-2026. */
package edu.kit.kastel.sdq.lissa.ratlr.artifactprovider;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import edu.kit.kastel.sdq.lissa.ratlr.configuration.ModuleConfiguration;
import edu.kit.kastel.sdq.lissa.ratlr.context.ContextStore;
import edu.kit.kastel.sdq.lissa.ratlr.knowledge.Artifact;

/**
 * Loads textual SysML v2 and KerML files as model artifacts.
 * The relative file path is retained as a stable artifact identifier so trace links
 * remain reproducible when a model is split across multiple packages or files.
 */
public final class SysMLV2ArtifactProvider extends ArtifactProvider {
    private final Path path;
    private final Set<String> extensions;
    private final List<Artifact> artifacts = new ArrayList<>();

    /**
     * Creates a provider.
     *
     * <p>Configuration arguments:
     * <ul>
     *   <li>{@code path}: model file or directory</li>
     *   <li>{@code extensions}: comma-separated extensions, default {@code sysml,kerml}</li>
     * </ul>
     *
     * @param configuration module configuration
     * @param contextStore shared pipeline context
     */
    public SysMLV2ArtifactProvider(ModuleConfiguration configuration, ContextStore contextStore) {
        super(contextStore);
        this.path = Path.of(configuration.argumentAsString("path"));
        this.extensions = Arrays.stream(configuration
                        .argumentAsString("extensions", "sysml,kerml")
                        .split(","))
                .map(String::trim)
                .filter(it -> !it.isEmpty())
                .map(it -> it.startsWith(".") ? it.toLowerCase(Locale.ROOT) : "." + it.toLowerCase(Locale.ROOT))
                .collect(Collectors.toUnmodifiableSet());
        if (!Files.exists(path)) {
            throw new IllegalArgumentException("Path does not exist: " + path.toAbsolutePath());
        }
    }

    @Override
    public List<Artifact> getArtifacts() {
        if (artifacts.isEmpty()) {
            loadArtifacts();
        }
        return List.copyOf(artifacts);
    }

    @Override
    public Artifact getArtifact(String identifier) {
        return getArtifacts().stream()
                .filter(it -> it.getIdentifier().equals(identifier))
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("Artifact not found: " + identifier));
    }

    private void loadArtifacts() {
        try (Stream<Path> stream = Files.isDirectory(path) ? Files.walk(path) : Stream.of(path)) {
            List<Path> files = stream.filter(Files::isRegularFile)
                    .filter(this::hasSupportedExtension)
                    .sorted()
                    .toList();
            if (files.isEmpty()) {
                throw new IllegalArgumentException(
                        "No SysML v2/KerML files with extensions " + extensions + " found under " + path);
            }
            for (Path file : files) {
                String identifier = Files.isDirectory(path)
                        ? path.relativize(file).toString().replace('\\', '/')
                        : file.getFileName().toString();
                artifacts.add(new Artifact(
                        identifier, Artifact.ArtifactType.SYSML_MODEL, Files.readString(file, StandardCharsets.UTF_8)));
            }
            artifacts.sort(Comparator.comparing(Artifact::getIdentifier));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private boolean hasSupportedExtension(Path file) {
        String fileName = file.getFileName().toString().toLowerCase(Locale.ROOT);
        return extensions.stream().anyMatch(fileName::endsWith);
    }
}

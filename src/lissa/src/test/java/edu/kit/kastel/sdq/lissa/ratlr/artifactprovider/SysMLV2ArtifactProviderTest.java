/* Licensed under MIT 2025-2026. */
package edu.kit.kastel.sdq.lissa.ratlr.artifactprovider;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import edu.kit.kastel.sdq.lissa.ratlr.configuration.ModuleConfiguration;
import edu.kit.kastel.sdq.lissa.ratlr.context.ContextStore;

class SysMLV2ArtifactProviderTest {
    @TempDir
    Path temporaryDirectory;

    @Test
    void loadsSupportedFilesRecursivelyWithStableIdentifiers() throws IOException {
        Files.writeString(temporaryDirectory.resolve("root.sysml"), "package Root {}");
        Path nested = Files.createDirectories(temporaryDirectory.resolve("library"));
        Files.writeString(nested.resolve("types.kerml"), "package Types {}");
        Files.writeString(nested.resolve("ignored.txt"), "not a model");

        var configuration = new ModuleConfiguration(
                "sysml_v2", Map.of("path", temporaryDirectory.toString(), "extensions", "sysml,kerml"));
        var provider = new SysMLV2ArtifactProvider(configuration, new ContextStore());

        var artifacts = provider.getArtifacts();
        assertEquals(2, artifacts.size());
        assertEquals("library/types.kerml", artifacts.get(0).getIdentifier());
        assertEquals("sysml model", artifacts.get(0).getType());
        assertEquals("root.sysml", provider.getArtifact("root.sysml").getIdentifier());
        assertThrows(IllegalArgumentException.class, () -> provider.getArtifact("missing.sysml"));
    }

    @Test
    void rejectsDirectoriesWithoutSupportedModelFiles() throws IOException {
        Files.writeString(temporaryDirectory.resolve("notes.txt"), "not a model");
        var configuration = new ModuleConfiguration(
                "sysml_v2", Map.of("path", temporaryDirectory.toString(), "extensions", "sysml,kerml"));
        var provider = new SysMLV2ArtifactProvider(configuration, new ContextStore());

        assertThrows(IllegalArgumentException.class, provider::getArtifacts);
    }
}

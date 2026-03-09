#include "vocal_model.h"
#include "vocal_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>

static char g_models_dir[4096];

const char * vocal_models_dir(const char * override_path) {
    if (override_path && override_path[0]) {
        snprintf(g_models_dir, sizeof(g_models_dir), "%s", override_path);
        return g_models_dir;
    }

    const char * env = getenv(VOCAL_ENV_MODELS_DIR);
    if (env && env[0]) {
        snprintf(g_models_dir, sizeof(g_models_dir), "%s", env);
        return g_models_dir;
    }

    const char * home = getenv("HOME");
    if (!home) {
        fprintf(stderr, "error: HOME not set\n");
        return NULL;
    }

    snprintf(g_models_dir, sizeof(g_models_dir), "%s/%s", home, VOCAL_DEFAULT_MODELS_DIR);
    return g_models_dir;
}

char * vocal_model_path(const char * model_name, const char * override_dir,
                        char * buf, int bufsize) {
    const char * dir = vocal_models_dir(override_dir);
    if (!dir) return NULL;

    int n = snprintf(buf, bufsize, "%s/%s", dir, model_name);
    if (n < 0 || n >= bufsize) return NULL;

    return buf;
}

bool vocal_model_exists(const char * model_name, const char * override_dir) {
    char path[4096];
    if (!vocal_model_path(model_name, override_dir, path, sizeof(path))) {
        return false;
    }

    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static int mkdirs(const char * path) {
    char tmp[4096];
    snprintf(tmp, sizeof(tmp), "%s", path);

    for (char * p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir(tmp, 0755);
            *p = '/';
        }
    }
    return mkdir(tmp, 0755);
}

int vocal_model_download(const char * url, const char * model_name,
                         const char * override_dir) {
    const char * dir = vocal_models_dir(override_dir);
    if (!dir) return 1;

    // Ensure directory exists
    mkdirs(dir);

    char path[4096];
    if (!vocal_model_path(model_name, override_dir, path, sizeof(path))) {
        fprintf(stderr, "error: path too long\n");
        return 1;
    }

    // Check if already downloaded
    struct stat st;
    if (stat(path, &st) == 0 && S_ISREG(st.st_mode)) {
        fprintf(stderr, "Model already exists: %s\n", path);
        return 0;
    }

    fprintf(stderr, "Downloading %s...\n", model_name);
    fprintf(stderr, "  URL: %s\n", url);
    fprintf(stderr, "  Destination: %s\n", path);

    // Use system curl with progress bar
    char cmd[8192];
    char tmp_path[4096];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path);
    snprintf(cmd, sizeof(cmd),
             "curl -L --progress-bar -o '%s' '%s'",
             tmp_path, url);

    int ret = system(cmd);
    if (ret != 0) {
        fprintf(stderr, "error: download failed (curl exit code %d)\n", ret);
        unlink(tmp_path);
        return 1;
    }

    // Move temp file to final path
    if (rename(tmp_path, path) != 0) {
        fprintf(stderr, "error: failed to move downloaded file\n");
        unlink(tmp_path);
        return 1;
    }

    // Verify file exists and has reasonable size
    if (stat(path, &st) != 0 || st.st_size < 1024) {
        fprintf(stderr, "error: downloaded file is too small or missing\n");
        unlink(path);
        return 1;
    }

    fprintf(stderr, "Downloaded: %s (%.1f MB)\n", path, st.st_size / (1024.0 * 1024.0));
    return 0;
}

void vocal_models_list(const char * override_dir) {
    const char * dir = vocal_models_dir(override_dir);
    if (!dir) {
        printf("No models directory found.\n");
        return;
    }

    DIR * d = opendir(dir);
    if (!d) {
        printf("No models downloaded yet.\n");
        printf("Run: vocal download asr\n");
        return;
    }

    printf("Models in %s:\n\n", dir);

    struct dirent * entry;
    int count = 0;
    while ((entry = readdir(d)) != NULL) {
        if (entry->d_name[0] == '.') continue;

        char path[4096];
        snprintf(path, sizeof(path), "%s/%s", dir, entry->d_name);

        struct stat st;
        if (stat(path, &st) == 0 && S_ISREG(st.st_mode)) {
            printf("  %-40s  %8.1f MB\n", entry->d_name, st.st_size / (1024.0 * 1024.0));
            count++;
        }
    }
    closedir(d);

    if (count == 0) {
        printf("  (none)\n");
        printf("\nRun: vocal download asr\n");
    }
    printf("\n");
}

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <inttypes.h>

void CoreSystemFloat32ToString(char *out, intptr_t outCount, float value)
{
    snprintf(out, outCount, "%f", value);
}

void CoreSystemFloat64ToString(char *out, intptr_t outCount, double value)
{
    snprintf(out, outCount, "%f", value);
}

void CoreSystemIntToString(char *out, intptr_t outCount, intptr_t value)
{
    snprintf(out, outCount, "%"PRIdPTR, value);
}

void CoreSystemUIntToString(char *out, intptr_t outCount, uintptr_t value)
{
    snprintf(out, outCount, "%"PRIuPTR, value);
}

void CoreSystemInt8ToString(char *out, intptr_t outCount, int8_t value)
{
    snprintf(out, outCount, "%"PRId8, value);
}

void CoreSystemUInt8ToString(char *out, intptr_t outCount, uint8_t value)
{
    snprintf(out, outCount, "%"PRIu8, value);
}

void CoreSystemInt16ToString(char *out, intptr_t outCount, int16_t value)
{
    snprintf(out, outCount, "%"PRId16, value);
}

void CoreSystemUInt16ToString(char *out, intptr_t outCount, uint16_t value)
{
    snprintf(out, outCount, "%"PRIu16, value);
}

void CoreSystemInt32ToString(char *out, intptr_t outCount, int32_t value)
{
    snprintf(out, outCount, "%"PRId32, value);
}

void CoreSystemUInt32ToString(char *out, intptr_t outCount, uint32_t value)
{
    snprintf(out, outCount, "%"PRIu32, value);
}

void CoreSystemInt64ToString(char *out, intptr_t outCount, int64_t value)
{
    snprintf(out, outCount, "%"PRId64, value);
}

void CoreSystemUInt64ToString(char *out, intptr_t outCount, uint64_t value)
{
    snprintf(out, outCount, "%"PRIu64, value);
}

void CoreSystemPointerToString(char *out, intptr_t outCount, void *value)
{
    snprintf(out, outCount, "%p", value);
}

void CoreSystemBoolToString(char *out, intptr_t outCount, bool value)
{
    snprintf(out, outCount, "%s", value ? "true" : "false");
}

void CoreSystemError(char *message)
{
    fflush(stdout);
    fprintf(stderr, "%s\n", message);
    abort();
}

void CoreSystemConsoleWriteChar(char c)
{
    putc(c, stdout);
}

void CoreSystemConsoleFlush(void)
{
    fflush(stdout);
}
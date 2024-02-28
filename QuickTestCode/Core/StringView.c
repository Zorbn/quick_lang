#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
void *memmove(void *dst, const void *src, size_t size);
int memcmp(const void *ptr1, const void *ptr2, size_t num);
int strcmp(const char *lhs, const char *rhs);

struct StringView__22 {
	char const *data;
	intptr_t count;
};

char __Core__StringView__StringView__Get(struct StringView__22 const *me, intptr_t index);

static inline bool StringView__22__Equals(struct StringView__22 *left, struct StringView__22 *right);

char __Core__StringView__StringView__Get(struct StringView__22 const *me, intptr_t index) {
	uintptr_t const dataStart = ((uintptr_t)me->data);
	uintptr_t const dataIndex = dataStart + ((uintptr_t)index) * sizeof(char);
	char const *const dataPointer = ((char const *)dataIndex);
	return (*dataPointer);
}

static inline bool StringView__22__Equals(struct StringView__22 *left, struct StringView__22 *right) {
	return left->data == right->data && left->count == right->count;
}


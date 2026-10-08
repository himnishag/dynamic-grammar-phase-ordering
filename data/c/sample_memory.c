#include <stdio.h>
int main() {
    unsigned int a = 5;
    unsigned int *p = &a;
    *p += 15;
    printf("%u\n", *p);
    return 0;
}

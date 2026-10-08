#include <stdio.h>
int main() {
    unsigned int x = 10, y = 20;
    unsigned int unused1 = x * 50;
    unsigned int unused2 = y + 100;
    (void)unused1; (void)unused2;
    printf("%u\n", x + y);
    return 0;
}

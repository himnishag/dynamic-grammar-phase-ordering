#include <stdio.h>
int main() {
    unsigned int sum = 0;
    for (unsigned int i = 0; i < 1000; i++) {
        sum += i * i;
    }
    printf("%u\n", sum);
    return 0;
}

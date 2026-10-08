#include <stdio.h>
int main() {
    unsigned int acc = 0;
    for (int i = 0; i < 50; i++) {
        for (int j = 0; j < 50; j++) {
            acc += (i + 1) * (j + 1);
        }
    }
    printf("%u\n", acc);
    return 0;
}

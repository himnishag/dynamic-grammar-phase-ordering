#include <stdio.h>
int main() {
    unsigned int val = 3, res = 0;
    switch (val & 3) {
        case 0: res = 10; break;
        case 1: res = 20; break;
        case 2: res = 30; break;
        default: res = 40; break;
    }
    printf("%u\n", res);
    return 0;
}

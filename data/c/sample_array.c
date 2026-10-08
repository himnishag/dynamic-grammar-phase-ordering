#include <stdio.h>
int main() {
    unsigned int arr[100];
    for (int i = 0; i < 100; i++) arr[i] = i * 3;
    unsigned int s = 0;
    for (int i = 0; i < 100; i++) s += arr[i];
    printf("%u\n", s);
    return 0;
}

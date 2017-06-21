/* vim: set sw=2 expandtab tw=80: */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <timer.h>
#include <console.h>
#include <gpio.h>
#include <led.h>
#include <tock.h>

// callback for timers
static void timer_cb (__attribute__ ((unused)) int arg0,
                      __attribute__ ((unused)) int arg1,
                      __attribute__ ((unused)) int arg2,
                      __attribute__ ((unused)) void* userdata) {}

// **************************************************
// GPIO output example
// **************************************************
static void gpio_output(void) {
  putstr("Periodically blinking LED\n");

  // Start repeating timer
  timer_every(500, timer_cb, NULL);

  while (1) {
    led_toggle(0);
    yield();
  }
}

// **************************************************
// GPIO input example
// **************************************************
static void gpio_input(void) {
  putstr("Periodically reading value of the GPIO 0 pin\n");
  putstr("Jump pin high to test (defaults to low)\n");

  // set LED pin as input and start repeating timer
  // pin is configured with a pull-down resistor, so it should read 0 as default
  gpio_enable_input(0, PullDown);
  timer_every(500, timer_cb, NULL);

  while (1) {
    // print pin value
    int pin_val = gpio_read(0);
    printf("\tValue(%d)\n", pin_val);
    yield();
  }
}

// **************************************************
// GPIO interrupt example
// **************************************************
static void gpio_cb (__attribute__ ((unused)) int pin_num,
                     __attribute__ ((unused)) int arg2,
                     __attribute__ ((unused)) int arg3,
                     __attribute__ ((unused)) void* userdata) {}

static void gpio_interrupt(void) {
  putstr("Print GPIO 0 pin reading whenever its value changes\n");
  putstr("Jump pin high to test\n");

  // set callback for GPIO interrupts
  gpio_interrupt_callback(gpio_cb, NULL);

  // set LED as input and enable interrupts on it
  gpio_enable_interrupt(0, PullDown, Change);

  while (1) {
    yield();
    putstr("\tGPIO Interrupt!\n");
  }
}


int main(void) {
  putstr("*********************\n");
  putstr("GPIO Test Application\n");

  // Set mode to which test you want
  uint8_t mode = 0;

  switch (mode) {
    case 0: gpio_interrupt(); break;
    case 1: gpio_output(); break;
    case 2: gpio_input(); break;
  }

  return 0;
}

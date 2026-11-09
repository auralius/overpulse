#pragma once
#include "../esp32_stub/Arduino.h"
#include <cstddef>
inline unsigned long millis() { return micros()/1000ul; }
using QueueHandle_t = void*;
struct portMUX_TYPE { int ignored; };
#define portMUX_INITIALIZER_UNLOCKED {0}
inline void portENTER_CRITICAL(portMUX_TYPE*) {}
inline void portEXIT_CRITICAL(portMUX_TYPE*) {}
#define pdTRUE 1
#define pdPASS 1
#define pdMS_TO_TICKS(x) (x)
inline QueueHandle_t xQueueCreate(unsigned, unsigned) { static int queue; return &queue; }
inline int xQueueSend(QueueHandle_t, const void*, unsigned) { return pdPASS; }
inline int xQueueReceive(QueueHandle_t, void*, unsigned) { return 0; }
inline void vTaskDelete(void*) {}
inline void vTaskDelay(unsigned) {}
inline int xTaskCreatePinnedToCore(void (*)(void*),const char*,int,void*,int,void*,int) { return pdPASS; }

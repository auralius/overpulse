#pragma once
#include <cstdint>
#define TFT_BLACK 0u
#define TFT_DARKGREY 1u
#define TFT_CYAN 2u
#define TFT_WHITE 3u
#define TFT_YELLOW 4u
#define TFT_RED 5u
#define TFT_GREEN 6u
class TFT_eSPI {
public:
 unsigned green_lines=0,red_lines=0,yellow_circles=0;
 int yellow_last_x=-1;
 int green_first_x=-1,red_first_x=-1;
 int green_last_x=-1,red_last_x=-1;
 int green_min_y=1000,green_max_y=-1000;
 int red_min_y=1000,red_max_y=-1000;
 void init() {}
 void setRotation(int) {}
 void fillScreen(uint32_t) {}
 void fillRect(int,int,int,int,uint32_t) {}
 void setTextColor(uint32_t,uint32_t) {}
 void setTextSize(int) {}
 void setCursor(int,int) {}
 void print(const char*) {}
 void drawRect(int,int,int,int,uint32_t) {}
 void drawFastHLine(int,int,int,uint32_t) {}
 void drawFastVLine(int,int,int,uint32_t) {}
 void drawLine(int x1,int y1,int x2,int y2,uint32_t color) {
  if(color==TFT_GREEN) {
   if(!green_lines) green_first_x=x1;
   green_last_x=x2;
   ++green_lines;
   if(y1<green_min_y)green_min_y=y1;
   if(y2<green_min_y)green_min_y=y2;
   if(y1>green_max_y)green_max_y=y1;
   if(y2>green_max_y)green_max_y=y2;
  } else if(color==TFT_RED) {
   if(!red_lines)red_first_x=x1;
   red_last_x=x2;
   ++red_lines;
   if(y1<red_min_y)red_min_y=y1;
   if(y2<red_min_y)red_min_y=y2;
   if(y1>red_max_y)red_max_y=y1;
   if(y2>red_max_y)red_max_y=y2;
  }
 }
 void fillCircle(int x,int,int,uint32_t color) {
  if(color==TFT_YELLOW) {++yellow_circles; yellow_last_x=x;}
 }
 template<typename... T>void printf(const char*,T...) {}
};

#define DELAY 40  //The default delay to check for a bounce is of 40 milliseconds
#define LED 6   //The first LED is on the digital pin n. 6
#define LED2 5 //The second LED is on the digital pin n. 5
#define BUTTON 7  //The measure of the button's state is on digital pin n. 7

typedef enum {
  BUTTON_UP,
  BUTTON_FALLING,
  BUTTON_DOWN,
  BUTTON_RAISING
} debounceState_t;

debounceState_t s;  //s refers to the current state


void debounceFSM_init(void);
void debounceFSM_update(void);

void pressed(void);
void released(void);

void setup() {
  pinMode(LED, OUTPUT);
  pinMode(BUTTON, INPUT);
  debounceFSM_init();
}

void loop() {
  debounceFSM_update();
}

void debounceFSM_init(void) {
  s = BUTTON_UP;
}

void debounceFSM_update(void) {
  static unsigned long last = millis();
 
  switch (s) {
    case BUTTON_UP:
      if (!digitalRead(BUTTON)){
        s = BUTTON_FALLING;
        last = millis();
      }
      break;

    case BUTTON_FALLING:
      if (millis() - last >= DELAY) {
        if (!digitalRead(BUTTON)){
          pressed();
          s = BUTTON_DOWN;
        }else s = BUTTON_UP;
      }
      break;

    case BUTTON_DOWN:
      if (digitalRead(BUTTON)) {
        s = BUTTON_RAISING;
        last = millis();
      }
      break;

    case BUTTON_RAISING:
      if (millis() - last >= DELAY) {
        if (digitalRead(BUTTON)) {
          released();
          s = BUTTON_UP;
        } else s = BUTTON_DOWN;
      }
      break;

    default:
      released();
      break;
  }
}

void pressed(void) {
  digitalWrite(LED, HIGH);
  digitalWrite(LED2, LOW);
}

void released(void) {
  digitalWrite(LED, LOW);
  digitalWrite(LED2, HIGH);
}

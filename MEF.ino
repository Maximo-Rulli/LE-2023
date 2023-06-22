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

debounceState_t state;  //state refers to the current state

bool on = HIGH;  //The variable on sets the state of the output LED that switches states

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
  state = BUTTON_UP;
}

void debounceFSM_update(void) {
  static unsigned long last = millis();
 
  switch (state) {
    case BUTTON_UP:
      if (!digitalRead(BUTTON)){
        state = BUTTON_FALLING;
        last = millis();
      }
      break;

    case BUTTON_FALLING:
      if (millis() - last >= DELAY) {
        if (!digitalRead(BUTTON)){
          pressed();
          state = BUTTON_DOWN;
        }else state = BUTTON_UP;
      }
      break;

    case BUTTON_DOWN:
      if (digitalRead(BUTTON)) {
        state = BUTTON_RAISING;
        last = millis();
      }
      break;

    case BUTTON_RAISING:
      if (millis() - last >= DELAY) {
        if (digitalRead(BUTTON)) {
          released();
          state = BUTTON_UP;
        } else state = BUTTON_DOWN;
      }
      break;

    default:
      released();
      break;
  }
}

void pressed(void) {
  digitalWrite(LED, HIGH);
  digitalWrite(LED2, on);
  on = !on;
}

void released(void) {
  digitalWrite(LED, LOW);
}

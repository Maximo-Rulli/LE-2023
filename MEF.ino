#define DELAY 40
#define LED 13
#define BUTTON 2

typedef enum {
  BUTTON_UP,
  BUTTON_FALLING,
  BUTTON_DOWN,
  BUTTON_RAISING
} S;

S s;


void debounceFSM_init(void);
void debounceFSM_update(void);

void buttonPressed(void);
void buttonReleased(void);

void setup() {
  pinMode(LED, OUTPUT);
  pinMode(BUTTON, INPUT);
  button_init();
}

void loop() {
  button_update();
}

void button_init(void) {
  s = BUTTON_UP;
}

void button_update(void) {
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
          buttonPressed();
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
          buttonReleased();
          s = BUTTON_UP;
        } else s = BUTTON_DOWN;
      }
      break;

    default:
      buttonReleased();
      break;
  }
}

void buttonPressed(void) {
  digitalWrite(LED, HIGH);
}

void buttonReleased(void) {
  digitalWrite(LED, LOW);
}

use simple_logger::SimpleLogger;
use std::sync::Once;

static INIT: Once = Once::new();

pub fn setup() {
    INIT.call_once(|| {
        SimpleLogger::new().init().unwrap();
    });
}
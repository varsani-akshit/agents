// While a Webull approval is pending, look again every 10 seconds so the page
// turns to "connected" by itself once you approve on your phone. Only loaded
// while waiting; a settled page never reloads under you.
setTimeout(() => location.replace("/portfolio"), 10000);

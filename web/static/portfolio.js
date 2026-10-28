// While something is pending — a Webull approval, a simulation run — look
// again every 10 seconds so the page updates by itself. Only loaded while
// waiting; a settled page never reloads under you.
setTimeout(() => location.replace(location.pathname + location.search), 10000);

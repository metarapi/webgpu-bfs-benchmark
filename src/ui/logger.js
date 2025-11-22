/**
 * Console logger utility for displaying color-coded log messages
 * in a terminal-style output element
 */
export class ConsoleLogger {
  constructor(elementId) {
    this.element = document.getElementById(elementId);
    this.maxLines = 100;
    this.lines = [];
    
    if (!this.element) {
      console.warn(`ConsoleLogger: element with id '${elementId}' not found`);
    }
  }

  /**
   * Log a message with specified level
   * @param {string} level - One of: info, success, warning, error, debug
   * @param {string} message - The message to log
   */
  log(level, message) {
    if (!this.element) return;

    const colors = {
      info: 'text-blue-400',
      success: 'text-green-400',
      warning: 'text-yellow-400',
      error: 'text-red-400',
      debug: 'text-purple-400'
    };

    const color = colors[level] || 'text-gray-400';
    const levelText = `[${String(level).toUpperCase()}]`;
    
    // Build HTML with Tailwind classes (no timestamp)
    const html = `<span class="${color}">${levelText}</span> ${message}`;
    
    this.lines.push(html);

    // Trim old lines if exceeding max
    if (this.lines.length > this.maxLines) {
      this.lines.splice(0, this.lines.length - this.maxLines);
    }

    // Update DOM and auto-scroll
    this.element.innerHTML = this.lines.map(l => l + '\n').join('');
    this.element.scrollTop = this.element.scrollHeight;
  }

  /**
   * Log info message (convenience method)
   */
  info(message) {
    this.log('info', message);
  }

  /**
   * Log success message (convenience method)
   */
  success(message) {
    this.log('success', message);
  }

  /**
   * Log warning message (convenience method)
   */
  warning(message) {
    this.log('warning', message);
  }

  /**
   * Log error message (convenience method)
   */
  error(message) {
    this.log('error', message);
  }

  /**
   * Log debug message (convenience method)
   */
  debug(message) {
    this.log('debug', message);
  }

  /**
   * Clear all log entries
   */
  clear() {
    if (!this.element) return;
    this.lines = [];
    this.element.innerHTML = '';
  }

  /**
   * Get current number of log lines
   */
  getLineCount() {
    return this.lines.length;
  }

  /**
   * Set maximum number of lines to keep
   */
  setMaxLines(max) {
    this.maxLines = max;
    if (this.lines.length > this.maxLines) {
      this.lines.splice(0, this.lines.length - this.maxLines);
      this.element.innerHTML = this.lines.map(l => l + '\n').join('');
    }
  }
}

/**
 * Create and return a global logger instance
 */
export function createLogger(elementId = 'console-output') {
  return new ConsoleLogger(elementId);
}

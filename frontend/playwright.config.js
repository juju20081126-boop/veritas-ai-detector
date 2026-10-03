// @ts-check
const { defineConfig, devices } = require('@playwright/test');
const path = require('path');
const fs = require('fs');

function findExistingChromium() {
  if (process.env.PLAYWRIGHT_CHROMIUM_PATH && fs.existsSync(process.env.PLAYWRIGHT_CHROMIUM_PATH)) {
    return process.env.PLAYWRIGHT_CHROMIUM_PATH;
  }
  const appData = process.env.LOCALAPPDATA || '';
  const candidate = path.join(appData, 'ms-playwright', 'chromium-1228', 'chrome-win64', 'chrome.exe');
  if (fs.existsSync(candidate)) {
    return candidate;
  }
  return undefined;
}

const execPath = findExistingChromium();

module.exports = defineConfig({
  testDir: './tests',
  testMatch: /.*\.spec\.(js|ts)$/,
  timeout: 30000,
  expect: {
    timeout: 5000,
  },
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  use: {
    baseURL: process.env.BASE_URL || 'http://127.0.0.1:8003',
    trace: 'off',
    screenshot: 'only-on-failure',
    launchOptions: execPath ? { executablePath: execPath } : {},
  },
  projects: [
    {
      name: 'chromium',
      use: {
        ...devices['Desktop Chrome'],
      },
    },
  ],
});

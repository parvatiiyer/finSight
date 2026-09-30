// @ts-check
const { test, expect } = require('@playwright/test');
const path = require('path');

const DASHBOARD_URL = `file://${path.resolve(__dirname, '../frontend/dashboard.html')}`;

test.describe('FinSight Dashboard E2E Suite', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(DASHBOARD_URL);
  });

  test('landing view displays properly and defaults to checking backend', async ({ page }) => {
    await expect(page.locator('#landingView')).toBeVisible();
    await expect(page.locator('#reportView')).toBeHidden();
    await expect(page.locator('#tickerInput')).toBeVisible();
    await expect(page.locator('#explainBtn')).toBeVisible();
    await expect(page.locator('#apiStatus')).toBeVisible();
  });

  test('offline fallback: click bundled example switches to report view with rendered decomposition and timeline', async ({ page }) => {
    // Click HDFC Bank example
    const hdfcBtn = page.locator('button[data-t="HDFCBANK.NS"]');
    await hdfcBtn.click();

    // Verify report view is now displayed
    await expect(page.locator('#reportView')).toBeVisible();
    await expect(page.locator('#landingView')).toBeHidden();

    // Verify stat strip rendered
    await expect(page.locator('.stat-strip')).toBeVisible();
    await expect(page.locator('#report')).toContainText('HDFCBANK.NS');

    // Verify decomposition bar and timeline rendered
    await expect(page.locator('.decomp-bar')).toBeVisible();
    await expect(page.locator('.timeline-svg')).toBeVisible();

    // Verify accepted and rejected evidence ledger
    await expect(page.locator('.ledger-cols')).toBeVisible();
    await expect(page.locator('.ledger-cols')).toContainText('Accepted evidence');
    await expect(page.locator('.ledger-cols')).toContainText('Considered, rejected');
  });

  test('new search button returns to landing view', async ({ page }) => {
    await page.locator('button[data-t="TCS.NS"]').click();
    await expect(page.locator('#reportView')).toBeVisible();

    await page.locator('#newSearchBtn').click();
    await expect(page.locator('#landingView')).toBeVisible();
    await expect(page.locator('#reportView')).toBeHidden();
  });

  test('offline fallback shows helpful message for unbundled tickers', async ({ page }) => {
    await page.locator('#tickerInput').fill('UNKNOWN_TICKER');
    await page.locator('#explainBtn').click();

    await expect(page.locator('#reportView')).toBeVisible();
    await expect(page.locator('#report')).toContainText('No live backend is running, so only the bundled examples work offline');
  });

  test('date range inputs update report header dates', async ({ page }) => {
    await page.locator('#startDate').fill('2025-01-01');
    await page.locator('#endDate').fill('2025-01-10');

    await page.locator('button[data-t="ICICIBANK.NS"]').click();
    await expect(page.locator('#reportStartDate')).toHaveText('2025-01-01');
    await expect(page.locator('#reportEndDate')).toHaveText('2025-01-10');
  });
});

import XCTest

@MainActor
final class StudioUITests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    private func launchStudio() -> XCUIApplication {
        let app = XCUIApplication()
        app.launchArguments = ["-studioEngine", "mlx"]
        app.launch()
        XCTAssertTrue(app.buttons["studio.engine-menu"].waitForExistence(timeout: 15))
        XCTAssertTrue(app.webViews.firstMatch.waitForExistence(timeout: 30))
        XCTAssertTrue(app.webViews.staticTexts["Create an asset"].waitForExistence(timeout: 40))
        XCTAssertTrue(app.webViews.staticTexts["Local MLX ready"].waitForExistence(timeout: 45), app.debugDescription)
        return app
    }

    func testLiveStudioControlsAndNativeEngineSwitch() {
        let app = launchStudio()
        XCTAssertTrue(app.buttons["studio.reload"].exists)
        app.buttons["studio.engine-menu"].tap()
        app.buttons["RTX 3060 · .54"].tap()
        XCTAssertTrue(app.buttons["studio.engine-menu"].waitForExistence(timeout: 10))
        app.buttons["studio.engine-menu"].tap()
        app.buttons["Mac · Local MLX"].tap()
        XCTAssertTrue(app.webViews.staticTexts["Create an asset"].waitForExistence(timeout: 30))
        XCTAssertTrue(app.webViews.staticTexts["Local MLX ready"].waitForExistence(timeout: 45), app.debugDescription)
        let faceLabel = app.webViews.descendants(matching: .any).matching(NSPredicate(format: "label == %@", "Preserve face")).firstMatch
        for _ in 0..<3 {
            if faceLabel.isHittable { break }
            app.webViews.firstMatch.swipeUp()
        }
        XCTAssertTrue(faceLabel.isHittable, app.debugDescription)
        faceLabel.tap()
        XCTAssertTrue(app.webViews.descendants(matching: .any).matching(NSPredicate(format: "label CONTAINS %@", "Face reference (optional)")).firstMatch.waitForExistence(timeout: 10))
    }

    func testGLBDownloadAppearsInNativeDownloads() {
        let app = launchStudio()
        let assetDownload = app.webViews.links.matching(NSPredicate(format: "label BEGINSWITH %@", "Download GLB")).firstMatch
        for _ in 0..<8 {
            if assetDownload.isHittable { break }
            app.webViews.firstMatch.swipeUp()
        }
        XCTAssertTrue(assetDownload.isHittable, app.debugDescription)
        let downloads = app.buttons["studio.downloads"]
        let previousLabel = downloads.label
        assetDownload.tap()
        let completed = NSPredicate(format: "label != %@", previousLabel)
        expectation(for: completed, evaluatedWith: downloads)
        waitForExpectations(timeout: 90)
        downloads.tap()
        XCTAssertTrue(app.staticTexts.matching(NSPredicate(format: "label ENDSWITH[c] %@", ".glb")).firstMatch.waitForExistence(timeout: 10))
    }

    func testNativePhotoPickerAddsAReferenceWithoutGenerating() {
        let app = launchStudio()
        let browse = app.webViews.buttons["browse files"]
        XCTAssertTrue(browse.isHittable)
        browse.tap()
        let library = app.buttons["Photo Library"]
        if library.waitForExistence(timeout: 5) { library.tap() }
        // The current system Photos picker exposes grid items as images, not cells.
        let photo = app.images.matching(identifier: "PXGGridLayout-Info").firstMatch
        XCTAssertTrue(photo.waitForExistence(timeout: 15), app.debugDescription)
        photo.tap()
        let done = app.buttons["Done"]
        XCTAssertTrue(done.waitForExistence(timeout: 5))
        XCTAssertTrue(done.isEnabled, app.debugDescription)
        done.tap()
        let secondReference = app.webViews.images.matching(NSPredicate(format: "label BEGINSWITH %@", "Reference view 2:")).firstMatch
        XCTAssertTrue(secondReference.waitForExistence(timeout: 15), app.debugDescription)
    }

    func testCompletedAssetRendersWithTexturedAndClayMaterials() {
        let app = launchStudio()
        let asset = app.webViews.switches["Preview Monkey King · face-compatible"]
        for _ in 0..<8 {
            if asset.isHittable { break }
            app.webViews.firstMatch.swipeUp()
        }
        XCTAssertTrue(asset.isHittable, app.debugDescription)
        asset.tap()
        let clay = app.webViews.switches["Clay"]
        expectation(for: NSPredicate(format: "enabled == true"), evaluatedWith: clay)
        waitForExpectations(timeout: 90)
        for _ in 0..<4 {
            if clay.isHittable { break }
            app.webViews.firstMatch.swipeDown()
        }
        XCTAssertTrue(clay.isHittable, app.debugDescription)
        clay.tap()
        XCTAssertEqual(clay.value as? String, "1")
        let textured = app.webViews.switches["Textured"]
        textured.tap()
        XCTAssertEqual(textured.value as? String, "1")
        let screenshot = XCTAttachment(screenshot: app.screenshot())
        screenshot.name = "Monkey King textured preview on iPhone"
        screenshot.lifetime = .keepAlways
        add(screenshot)
    }

}

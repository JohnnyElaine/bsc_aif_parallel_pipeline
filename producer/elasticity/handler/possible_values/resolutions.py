from producer.data.resolution import Resolution


class AllResolutions:
    # Common 4:3 resolutions
    RATIO_4_3 = [
        Resolution(640, 480),    # VGA
        Resolution(800, 600),    # SVGA
        Resolution(1024, 768),   # XGA
        Resolution(1280, 960),  # SXGA
        Resolution(1400, 1050),  # SXGA+
        Resolution(1600, 1200),  # UXGA
        Resolution(2048, 1536)   # QXGA
    ]

    # Common 16:9 resolutions
    RATIO_16_9 = [
        Resolution(1024, 576),   # WSVGA
        Resolution(1152, 648),  # HD-ready
        Resolution(1280, 720),  # HD (720p)
        Resolution(1366, 768),  # WXGA
        Resolution(1600, 900),  # HD+
        Resolution(1920, 1080), # Full HD (1080p)
        Resolution(2560, 1440), # QHD (1440p)
        Resolution(3840, 2160), # 4K UHD
        Resolution(7680, 4320)  # 8K UHD
    ]
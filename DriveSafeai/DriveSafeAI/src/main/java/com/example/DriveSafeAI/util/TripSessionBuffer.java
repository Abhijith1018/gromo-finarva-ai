package com.example.DriveSafeAI.util;

import com.example.DriveSafeAI.dto.LiveTripDTO;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class TripSessionBuffer {
    private static final Map<String, List<LiveTripDTO>> sessionMap = new ConcurrentHashMap<>();

    public static void addToSession(String sessionId, LiveTripDTO dto) {
        sessionMap.computeIfAbsent(sessionId, k -> new ArrayList<>()).add(dto);
    }

    public static List<LiveTripDTO> endSession(String sessionId) {
        return sessionMap.remove(sessionId); // Removes and returns
    }

    public static boolean hasSession(String sessionId) {
        return sessionMap.containsKey(sessionId);
    }
}

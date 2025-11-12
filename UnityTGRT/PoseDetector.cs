using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Reflection;
using TMPro;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.XR.Hands.Samples.GestureSample; // StaticHandGesture

public class PoseDetector : MonoBehaviour
{
    [Serializable]
    public struct GestureBinding
    {
        [Tooltip("XR Hands sample gesture component that raises performed/ended events")]
        public StaticHandGesture gesture;

        [Tooltip("Numeric label to use in logs/datasets (e.g., 0=rest, 1=pinch, 2=point, 3=fist, ...)")]
        public int label;

        [Tooltip("Optional display name for UI/debug")]
        public string displayName;
    }

    [Header("Bindings")]
    [Tooltip("Map each StaticHandGesture to its numeric label")]
    public List<GestureBinding> bindings = new List<GestureBinding>();

    [Header("Defaults")]
    [Tooltip("Label value used when no gesture is active")]
    public int restLabel = 0;

    [Header("Optional UI")]
    public TextMeshProUGUI gestureText;

    // Internal
    private BBH_eRead bbhERead;
    private int currentLabel = int.MinValue;  // “unset” at start

    // Cached reflection (optional) for new logging API
    private MethodInfo miLogLabelChangeFromHost; // (int label, long hostUs)
    private MethodInfo miUpdateGestureString;    // legacy: updateGesture(string)

    // High-res host clock
    private static readonly double TICK_TO_US = 1_000_000.0 / Stopwatch.Frequency;
    private static long HostNowUs() => (long)(Stopwatch.GetTimestamp() * TICK_TO_US);

    void Awake()
    {
        bbhERead = GameObject.FindObjectOfType<BBH_eRead>();
        if (bbhERead != null)
        {
            miLogLabelChangeFromHost = bbhERead.GetType().GetMethod(
                "LogLabelChangeFromHost",
                BindingFlags.Public | BindingFlags.Instance,
                null,
                new[] { typeof(int), typeof(long) },
                null);

            miUpdateGestureString = bbhERead.GetType().GetMethod(
                "updateGesture",
                BindingFlags.Public | BindingFlags.Instance,
                null,
                new[] { typeof(string) },
                null);
        }

        // Initialize UI with rest
        SetUi(restLabel);
    }

    void OnEnable()
    {
        // Subscribe (with closures) to avoid capturing loop variable by ref
        foreach (var b in bindings)
        {
            if (b.gesture == null) continue;
            int lbl = b.label;
            string name = string.IsNullOrEmpty(b.displayName) ? b.gesture.gameObject.name : b.displayName;

            b.gesture.gesturePerformed.AddListener(() => OnGesturePerformed(lbl, name));
            b.gesture.gestureEnded   .AddListener(() => OnGestureEnded(lbl, name));
        }
    }

    void OnDisable()
    {
        // Unsubscribe to avoid duplicate handlers on re-enable
        foreach (var b in bindings)
        {
            if (b.gesture == null) continue;
            b.gesture.gesturePerformed.RemoveAllListeners();
            b.gesture.gestureEnded.RemoveAllListeners();
        }
    }

    private void OnGesturePerformed(int label, string displayName)
    {
        // If the same label is already active, ignore (debounce)
        if (label == currentLabel) return;

        // If another gesture is active, treat this as a switch
        SetLabel(label, displayName);
    }

    private void OnGestureEnded(int label, string displayName)
    {
        // Only drop to rest if the ending gesture is the one we think is active;
        // if a different gesture replaced it, ignore this "ended".
        if (currentLabel != label) return;

        SetLabel(restLabel, "Rest");
    }

    private void SetLabel(int newLabel, string displayName)
    {
        currentLabel = newLabel;

        // 1) Update UI
        SetUi(newLabel, displayName);

        // 2) Emit label change with timestamp
        long hostUs = HostNowUs();

        // Preferred path: let BBH_eRead convert host->device and write "t_us|LABEL|k"
        if (bbhERead != null && miLogLabelChangeFromHost != null)
        {
            try { miLogLabelChangeFromHost.Invoke(bbhERead, new object[] { newLabel, hostUs }); }
            catch (Exception e) { UnityEngine.Debug.LogWarning($"LogLabelChangeFromHost failed: {e.Message}"); }
        }

        // Back-compat: still tell the old pipeline about the current label (string)
        if (bbhERead != null && miUpdateGestureString != null)
        {
            try { miUpdateGestureString.Invoke(bbhERead, new object[] { newLabel.ToString() }); }
            catch (Exception e) { UnityEngine.Debug.LogWarning($"updateGesture failed: {e.Message}"); }
        }
    }

    private void SetUi(int label, string nameOverride = null)
    {
        if (gestureText == null) return;

        if (label == restLabel)
            gestureText.text = restLabel.ToString();
        else
            gestureText.text = string.IsNullOrEmpty(nameOverride) ? label.ToString() : nameOverride;
    }
}

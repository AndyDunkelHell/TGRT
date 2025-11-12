using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Text;
using System.Threading;
using UnityEngine;
using System.Collections.Concurrent;

public class BBH_eRead : MonoBehaviour
{
    // ---------------- Public config ----------------

    public bool eDataBool = false;
    public bool ReadFromBoard = false;

    private bool readLoop = false;
    public bool ReadFromFile = false;

    [Tooltip("SerialPort of your device.")]
    public string portName = "COM8";

    [Tooltip("Baudrate")]
    public int baudRate = 250000;

    [Tooltip("QueueLength")]
    public int QueueLength = 1;

    [Tooltip("Index suffix for log file names")]
    public int LogIndex = 0;

    [Header("UDP (LiveMonitor)")]
    public int listenPort = 5555;

    [Header("Label logging")]
    public bool writeLabelLog = true;

    // Current label (string, for legacy UI / compatibility)
    public string gesture = "0";

    // Last parsed values (for your other components)
    public float[] eValues   = new float[] {0,0,0,0,0,0,0,0,0,0,0,0};
    public float[] IMUvalues = new float[] {0,0,0,0,0,0};

    // --------------- Private fields ----------------
    BBH myDevice = new BBH(); // Your board wrapper

    private TKE_MAfinal Stream; // your existing processing script

    // File paths
    private string dataPath;
    private string labelPath;
    private readonly object fileLock  = new object();
    private readonly object labelLock = new object();

    // UDP thread
    private Thread udpThread;
    private volatile bool udpRun = false;
    private UdpClient udp;
    private IPEndPoint remoteEP = new IPEndPoint(IPAddress.Any, 0);

    // Clock mapping host_us -> device_us
    private ClockMapper clockMap = new ClockMapper(2048);

    // High-res host clock
    private static readonly double TICK_TO_US = 1_000_000.0 / Stopwatch.Frequency;
    private static long HostNowUs() => (long)(Stopwatch.GetTimestamp() * TICK_TO_US);

    // For fallback timestamping if no device ts exists
    private static readonly DateTime _unixEpoch = new DateTime(1970,1,1,0,0,0, DateTimeKind.Utc);

    // --- NEW: serial reader thread + queue ---
    private Thread serialThread;
    private volatile bool serialRun = false;
    private readonly ConcurrentQueue<string> serialUiQueue = new ConcurrentQueue<string>();

    // ===== Label + time-sync state =====
    private volatile int _currentLabel = 0;              // default "rest"
    [SerializeField] private bool includeLabelInRows = true; // put label into each data row

    // host<->device time mapping (EWMA of host_us - device_us from incoming frames)
    private long devHostOffsetUs = 0;
    private int  devHostOffsetSamples = 0;
    private long HostToDeviceUs(long hostUs)
    {
        // Use mapped time once we've seen enough device timestamps to stabilize the EWMA
        if (devHostOffsetSamples >= 20) return hostUs - devHostOffsetUs;
        return hostUs; // bootstrap period
    }

    public int GetCurrentLabel() => Volatile.Read(ref _currentLabel); // optional helper

    // ---------------- Unity lifecycle ----------------
    void Start()
    {
        // Ensure output dir exists
        string outDir = Path.Combine(Application.dataPath, "CUSTOM", "BBH_ELECTRODES");
        try { Directory.CreateDirectory(outDir); } catch {}

        dataPath  = Path.Combine(outDir, $"eLog{LogIndex}.txt");
        labelPath = Path.Combine(outDir, $"labelLog{LogIndex}.txt");

        // Optional: fresh files per run
        try { File.WriteAllText(dataPath,  ""); } catch {}
        try { File.WriteAllText(labelPath, ""); } catch {}

        // TKE-MA component hookup
        Stream = GameObject.FindObjectOfType<TKE_MAfinal>();


        if (ReadFromFile)
        {
            try
            {
                udp = new UdpClient(listenPort);
                udpRun = true;
                udpThread = new Thread(UdpPump) { IsBackground = true, Name = "BBH_eRead_UDP" };
                udpThread.Start();
                UnityEngine.Debug.Log($"<color=orange>* UDP tail on port {listenPort} (threaded)</color>");
            }
            catch (Exception ex)
            {
                UnityEngine.Debug.LogError($"[BBH_eRead] Failed to start UDP: {ex.Message}");
                ReadFromFile = false;
            }
        }
    }

    public void Connect()
    {
        if (ReadFromBoard)
        {
            myDevice.set(portName, baudRate);
            myDevice.connect();
        }

        myDevice.send("!identity");
        readLoop = true;

        // --- NEW: spin up background serial pump ---
        if (ReadFromBoard && serialThread == null)
        {
            serialRun = true;
            serialThread = new Thread(SerialPump) { IsBackground = true, Name = "BBH_eRead_SERIAL" };
            serialThread.Start();
            UnityEngine.Debug.Log("<color=orange>* Serial pump started (threaded)</color>");
        }
    }

    public void StartEStream() { myDevice.send("!connect");  }
    public void Disconnect()   { myDevice.send("!DC");       }

    void Update()
    {
        // nothing to do if not reading from the board
        if (ReadFromBoard)
        {
            // Drain up to a few packets per frame to keep UI responsive
            int drained = 0;
            while (drained < 8 && serialUiQueue.TryDequeue(out var line))
            {
                // line is "emgCSV|imuCSV"
                UpdateElectrode(line);
                drained++;
            }
        }

    }

    void OnApplicationQuit()
    {
                // stop serial thread
        if (serialThread != null)
        {
            serialRun = false;
            try { serialThread.Join(200); } catch {}
            serialThread = null;
        }
        // stop UDP thread
        if (udpRun)
        {
            udpRun = false;
            try { udp?.Close(); } catch { }
            try
            {
                if (udpThread != null && udpThread.IsAlive)
                    udpThread.Join(200);
            }
            catch { }
        }

        if (ReadFromBoard)
        {
            try { myDevice.send("!DC"); } catch {}
            try { myDevice.close(); }    catch {}
        }
    }

    // --- NEW: background serial reader ---
    // Reads myDevice queue continuously, logs to file, pushes compact "emg|imu" to the UI queue.
    private void SerialPump()
    {
        // High-res host fallback if device ts missing
        while (serialRun)
        {
            try
            {
                // Non-blocking poll of the wrapper queue
                string msg = myDevice.readQueue();
                if (msg == null) { Thread.Sleep(1); continue; }

                // Parse variants: emg|imu  |  t|emg|imu  |  t|seq|emg|imu
                string[] p = msg.Split('|');
                long devUs = -1;
                string seq = "";
                string emgCsv = "", imuCsv = "";

                if (p.Length >= 4 && long.TryParse(p[0], out devUs))
                {   // t|seq|emg|imu
                    seq = p[1]; emgCsv = p[2]; imuCsv = p[3];
                }
                else if (p.Length == 3 && long.TryParse(p[0], out devUs))
                {   // t|emg|imu
                    emgCsv = p[1]; imuCsv = p[2];
                }
                else if (p.Length == 2)
                {   // emg|imu
                    emgCsv = p[0]; imuCsv = p[1];
                }
                else
                {
                    continue; // malformed
                }

            long hostNow = HostNowUs();

            // --- (a) Learn offset when a device timestamp is present
            if (devUs >= 0)
            {
                long sampleOffset = hostNow - devUs; // host_us - device_us
                devHostOffsetUs = (devHostOffsetSamples == 0)
                    ? sampleOffset
                    : (long)(0.98 * devHostOffsetUs + 0.02 * sampleOffset);
                if (devHostOffsetSamples < 1_000_000) devHostOffsetSamples++;
            }

            long tsOut = (devUs >= 0) ? devUs : HostToDeviceUs(hostNow);

            // --- (b) Embed the current label into every data row (optional but recommended)
            string outLine;
            if (includeLabelInRows)
            {
                int lbl = Volatile.Read(ref _currentLabel);
                // t_us|label|EMG_CSV|IMU_CSV|host_us
                outLine = $"{tsOut}|{lbl}|{emgCsv}|{imuCsv}|{hostNow}\n";
            }
            else
            {
                // legacy: t_us|EMG_CSV|IMU_CSV|host_us
                outLine = $"{tsOut}|{emgCsv}|{imuCsv}|{hostNow}\n";
            }

            try { lock (fileLock) System.IO.File.AppendAllText(dataPath, outLine); } catch {}

            // hand to UI/update path as before (no Unity calls off-thread)
            serialUiQueue.Enqueue($"{emgCsv}|{imuCsv}");
        }
        catch { /* keep pumping */ }
    }
}

    // ---------------- UDP background pump ----------------
    private void UdpPump()
    {
        // Blocking Receive loop
        while (udpRun)
        {
            try
            {
                byte[] packet = udp.Receive(ref remoteEP);     // blocks until a datagram arrives
                long hostNow = HostNowUs();
                string line = Encoding.ASCII.GetString(packet);

                // Accept:
                //   emg|imu
                //   t_us|emg|imu
                //   t_us|seq|emg|imu
                string[] p = line.Split('|');

                long devUs = -1;
                string seq = "";
                string emgCsv, imuCsv;

                if (p.Length >= 4 && long.TryParse(p[0], out devUs))
                {
                    // t|seq|emg|imu
                    seq = p[1];
                    emgCsv = p[2];
                    imuCsv = p[3];
                }
                else if (p.Length == 3 && long.TryParse(p[0], out devUs))
                {
                    // t|emg|imu
                    emgCsv = p[1];
                    imuCsv = p[2];
                }
                else if (p.Length == 2)
                {
                    // emg|imu (legacy, no device time)
                    emgCsv = p[0];
                    imuCsv = p[1];
                }
                else
                {
                    continue; // malformed line
                }

                // Warm up clock mapper when device time is present
                if (devUs >= 0)
                {
                    clockMap.Add(hostNow, devUs);
                }

                // Write canonical data log
                try
                {
                    // Prefer device time; fallback to host microseconds if not available
                    long tsOut = (devUs >= 0) ? devUs : hostNow;

                    string outLine;
                    if (!string.IsNullOrEmpty(seq))
                        outLine = $"{tsOut}|{seq}|{emgCsv}|{imuCsv}|{hostNow}\n";
                    else
                        outLine = $"{tsOut}|{emgCsv}|{imuCsv}|{hostNow}\n";

                    lock (fileLock) File.AppendAllText(dataPath, outLine);
                }
                catch { /* ignore I/O hiccups */ }
                UpdateElectrode($"{emgCsv}|{imuCsv}");
            }
            catch (SocketException)
            {
                if (!udpRun) break;
            }
            catch (ObjectDisposedException)
            {
                break;
            }
            catch (Exception ex)
            {
                UnityEngine.Debug.LogWarning($"[BBH_eRead] UDP pump error: {ex.Message}");
            }
        }
    }

    // ---------------- Label logging API ----------------
    /// <summary>
    /// Called by PoseDetector (or similar). hostUs = Unity high-res microseconds.
    /// This converts host->device time and appends:  t_us|LABEL|<label>
    /// </summary>
    public void LogLabelChangeFromHost(int label, long hostUs)
    {
        if (!writeLabelLog) return;

        // Update the “current label” atomically so the data pump can embed it
        Interlocked.Exchange(ref _currentLabel, label);

        long tDevUs = HostToDeviceUs(hostUs);

        // Format: t_us|LABEL|<label>|<host_us>
        string line = $"{tDevUs}|LABEL|{label}|{hostUs}\n";

        try
        {
            lock (labelLock)
            {
                if (string.IsNullOrEmpty(labelPath))
                {
                    string outDir = Path.Combine(Application.dataPath, "CUSTOM", "BBH_ELECTRODES");
                    labelPath = Path.Combine(outDir, $"labelLog{LogIndex}.txt");
                }
                File.AppendAllText(labelPath, line);
            }
        }
        catch (Exception ex)
        {
            UnityEngine.Debug.LogWarning($"[BBH_eRead] label log write failed: {ex.Message}");
        }
    }
    public void updateGesture(string g)
    {
        gesture = g;
    }

    public void UpdateElectrode(string eInput)
    {
        try
        {
            // Expected "emgCSV|imuCSV"
            string[] splitInput = eInput.Split('|');
            if (splitInput.Length < 2) return;

            float[] inEArray   = Array.ConvertAll(splitInput[0].Split(','), float.Parse);
            float[] inIMUArray = Array.ConvertAll(splitInput[1].Split(','), float.Parse);

            if (inEArray.Length == 12)
            {
                if (inEArray[1] > 5)
                {
                    eValues   = inEArray;
                    IMUvalues = inIMUArray;

                    // (already done in Update() for serial; UDP path may not need it)
                    // Stream?.eDataAppend(eInput);
                }
            }
        }
        catch
        {
            // swallow parse errors to keep UI responsive
        }
    }

    // ---------------- Online linear fit ----------------
    /// <summary>
    /// Maintains a rolling window least-squares fit for device_us ≈ a*host_us + b
    /// to convert Unity's high-res timestamps into the board's device timebase.
    /// </summary>
    // ==================== Clock mapper helper (nested) ====================
    sealed class ClockMapper
    {
        // device_us ≈ a * host_us + b over a rolling window
        private readonly int capacity;
        private readonly Queue<(double host, double dev)> buf = new Queue<(double, double)>();
        private double sumH=0, sumD=0, sumHH=0, sumHD=0;
        private double a = 1.0, b = 0.0;
        private readonly object mtx = new object();

        public ClockMapper(int window = 1024)
        {
            capacity = Math.Max(32, window);
        }

        public void Add(long host_us, long dev_us)
        {
            lock (mtx)
            {
                double h = host_us, d = dev_us;
                buf.Enqueue((h, d));
                sumH += h; sumD += d; sumHH += h*h; sumHD += h*d;

                if (buf.Count > capacity)
                {
                    var (oh, od) = buf.Dequeue();
                    sumH -= oh; sumD -= od; sumHH -= oh*oh; sumHD -= oh*od;
                }

                int n = buf.Count;
                double denom = n * sumHH - sumH * sumH;
                if (Math.Abs(denom) > 1e-9)
                {
                    a = (n * sumHD - sumH * sumD) / denom;
                    b = (sumD - a * sumH) / n;
                }
                // else keep previous a,b
            }
        }

        public long HostToDevice(long host_us)
        {
            lock (mtx)
            {
                return (long)(a * host_us + b);
            }
        }

        public bool TryHostToDevice(long host_us, out long device_us)
        {
            lock (mtx)
            {
                if (buf.Count < 32) { device_us = host_us; return false; }
                device_us = (long)(a * host_us + b);
                return true;
            }
        }
    }


}


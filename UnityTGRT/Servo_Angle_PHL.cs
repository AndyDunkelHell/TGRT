using System.IO;
using UnityEngine;

public class Servo_Angle_PHL : MonoBehaviour
{
    [Header("Joint Transforms")]
    public Transform palmTransform;    // assign the GameObject named "Palm"
    public Transform originTransform;  // e.g. "IndexMetacarpal"
    public Transform trackTransform;   // e.g. "IndexProximal"

    [Header("Servo Settings")]
    public int ServoIndex = 0;
    public int Adjustment = 0;

    [Header("Live Debug")]
    [SerializeField] private float currentAngle = 0f;
    public float CurrentAngle => currentAngle;

    private int prevAngle = int.MinValue;
    private BBHWrite BBHWrite;

    void Start()
    {
        BBHWrite = FindObjectOfType<BBHWrite>();
        // ensure log exists
        var path = Application.dataPath + $"/CUSTOM/mLog{ServoIndex}.txt";
        if (!File.Exists(path)) File.WriteAllText(path, "");
    }

    void Update()
    {
        Vector3 pivot = originTransform.position;               // e.g. knuckle joint
        Vector3 v1    = palmTransform.position  - pivot;       // vector from joint back into the palm
        Vector3 v2    = trackTransform.position - pivot;       // vector from joint up the finger

        // 2) normalize and measure their angle
        v1.Normalize();
        v2.Normalize();
        float angleF = Vector3.Angle(v1, v2);                  // 0°…180°, always positive

        

        int angleI = Mathf.RoundToInt(angleF) + Adjustment;
        currentAngle = angleI;
        if (angleI != prevAngle && Mathf.Abs(angleI - prevAngle) > 2)
        {
            BBHWrite.UpdateAngle(angleI, ServoIndex);
            prevAngle = angleI;
            File.AppendAllText(
                Application.dataPath + $"/CUSTOM/mLog{ServoIndex}.txt",
                $"{Time.time};{angleF:F2}\n"
            );
        }

        Debug.DrawLine(originTransform.position, trackTransform.position, Color.cyan);
    }
}

// Servo_Angle_PHL_Editor.cs
// No changes required here—there were no Oculus dependencies.
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System.IO;
 
[CustomEditor(typeof(Servo_Angle_PHL))]
public class Servo_Angle_PHL_Editor : Editor
{
    Servo_Angle_PHL targetManager;
    private static int selec;

    private void OnEnable() {
        targetManager = (Servo_Angle_PHL)target;
    }

    public override void OnInspectorGUI()
    {
        base.OnInspectorGUI();

        Servo_Angle_PHL script = (Servo_Angle_PHL)target;

        if(GUILayout.Button("Start Recording Animation", GUILayout.Height(35)))
        {
            string path = Application.dataPath + "/CUSTOM/mLog"+ script.ServoIndex + ".txt";
            Debug.Log("Creating Log "+ script.ServoIndex);
            selec = script.ServoIndex;    
            
            if (!File.Exists(path)){
                File.WriteAllText(path,"\n\n");
            }    
        }
    }
}

using UnityEngine;
using System.Diagnostics;
using System.IO;

public class PythonLauncher : MonoBehaviour
{
    // Nombre o ruta relativa del script Python dentro de Assets
    public string pythonScriptRelativePath = "PythonScripts/propuesta_ROSario.v2.py";

    // Opcional: ruta al ejecutable Python. Si python está en PATH, puedes solo usar "python"
    public string pythonExe = "python";

    void Start()
    {
        RunPythonScript();
    }

    void RunPythonScript()
    {
        string scriptFullPath = Path.Combine(Application.dataPath, pythonScriptRelativePath);

        ProcessStartInfo start = new ProcessStartInfo
        {
            FileName = pythonExe,
            Arguments = $"\"{scriptFullPath}\"",
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true
        };

        try
        {
            Process process = new Process();
            process.StartInfo = start;

            process.OutputDataReceived += (sender, e) => {
                if (!string.IsNullOrEmpty(e.Data))
                    UnityEngine.Debug.Log("Salida Python: " + e.Data);
            };

            process.ErrorDataReceived += (sender, e) => {
                if (!string.IsNullOrEmpty(e.Data))
                    UnityEngine.Debug.LogError("Error Python: " + e.Data);
            };

            process.Start();
            process.BeginOutputReadLine();
            process.BeginErrorReadLine();

            // No llamamos a WaitForExit, para evitar congelamiento
        }
        catch (System.Exception e)
        {
            UnityEngine.Debug.LogError("Error al ejecutar Python: " + e.Message);
        }
    }

}

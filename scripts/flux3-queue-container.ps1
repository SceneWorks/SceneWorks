# Owned per-campaign process container, not runner/task administration. Normal user privileges.
# Assign the suspended child to a nested kill-on-close Job BEFORE its first instruction. Only this
# helper holds the job handle; if it dies, Windows terminates the complete owned descendant tree.
[CmdletBinding(DefaultParameterSetName='run')]
param([Parameter(Mandatory=$true,ParameterSetName='run')][string]$BootFile,
      [Parameter(Mandatory=$true,ParameterSetName='query')][string]$QueryJob)
$ErrorActionPreference='Stop'
Add-Type -TypeDefinition @'
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using System.Security.Cryptography;
public sealed class Flux3OwnedContainer : IDisposable {
 [StructLayout(LayoutKind.Sequential)] struct SECURITY { public int length; public IntPtr descriptor; public int inherit; }
 [StructLayout(LayoutKind.Sequential)] struct STARTUP { public int cb; public string reserved, desktop, title; public int x,y,xSize,ySize,xCount,yCount,fill,flags; public short show,reserved2; public IntPtr reservedPtr,input,output,error; }
 [StructLayout(LayoutKind.Sequential)] struct PROCESS { public IntPtr process,thread; public uint pid,tid; }
 [StructLayout(LayoutKind.Sequential)] struct BASIC_LIMIT { public long perProcess,perJob; public uint flags; public UIntPtr min,max; public uint active; public UIntPtr affinity; public uint priority,scheduling; }
 [StructLayout(LayoutKind.Sequential)] struct IO { public ulong readOp,writeOp,otherOp,readBytes,writeBytes,otherBytes; }
 [StructLayout(LayoutKind.Sequential)] struct LIMIT { public BASIC_LIMIT basic; public IO io; public UIntPtr processMemory,jobMemory,peakProcess,peakJob; }
 [StructLayout(LayoutKind.Sequential)] struct ACCOUNTING { public long user,kernel,periodUser,periodKernel; public uint faults,total,active,terminated; }
 [DllImport("kernel32.dll",CharSet=CharSet.Unicode,SetLastError=true)] static extern IntPtr CreateJobObject(IntPtr attr,string name);
 [DllImport("kernel32.dll",CharSet=CharSet.Unicode,SetLastError=true)] static extern IntPtr OpenJobObject(uint access,bool inherit,string name);
 [DllImport("kernel32.dll",SetLastError=true)] static extern bool SetInformationJobObject(IntPtr job,int kind,ref LIMIT info,uint length);
 [DllImport("kernel32.dll",SetLastError=true)] static extern bool QueryInformationJobObject(IntPtr job,int kind,ref ACCOUNTING info,uint length,IntPtr returned);
 [DllImport("kernel32.dll",SetLastError=true)] static extern bool AssignProcessToJobObject(IntPtr job,IntPtr process);
 [DllImport("kernel32.dll",SetLastError=true)] static extern bool TerminateJobObject(IntPtr job,uint code);
 [DllImport("kernel32.dll",SetLastError=true)] static extern bool TerminateProcess(IntPtr process,uint code);
 [DllImport("kernel32.dll",SetLastError=true)] static extern uint ResumeThread(IntPtr thread);
 [DllImport("kernel32.dll",SetLastError=true)] static extern uint WaitForSingleObject(IntPtr handle,uint ms);
 [DllImport("kernel32.dll",SetLastError=true)] static extern bool GetExitCodeProcess(IntPtr process,out uint code);
 [DllImport("kernel32.dll",SetLastError=true)] static extern bool CloseHandle(IntPtr handle);
 [DllImport("kernel32.dll",CharSet=CharSet.Unicode,SetLastError=true)] static extern bool CreateProcess(string exe,StringBuilder command,IntPtr pa,IntPtr ta,bool inherit,uint flags,IntPtr environment,string cwd,ref STARTUP startup,out PROCESS process);
 [DllImport("kernel32.dll",CharSet=CharSet.Unicode,SetLastError=true)] static extern IntPtr CreateFile(string name,uint access,uint share,ref SECURITY security,uint creation,uint flags,IntPtr template);
 IntPtr job,child; RSACryptoServiceProvider key; Process owner; long ownerStarted; bool stopping; uint exitCode=1; public uint ChildPid {get;private set;}
 static void Need(bool ok,string action) { if(!ok) throw new InvalidOperationException(action+" failed: "+Marshal.GetLastWin32Error()); }
 public Flux3OwnedContainer(string name,int ownerPid) {
  owner=Process.GetProcessById(ownerPid); ownerStarted=owner.StartTime.ToUniversalTime().Ticks;
  job=CreateJobObject(IntPtr.Zero,name); Need(job!=IntPtr.Zero,"CreateJobObject");
  if(Marshal.GetLastWin32Error()==183) {CloseHandle(job);job=IntPtr.Zero;throw new InvalidOperationException("Existing container refused");}
  try { var limit=new LIMIT(); limit.basic.flags=0x2000; Need(SetInformationJobObject(job,9,ref limit,(uint)Marshal.SizeOf(typeof(LIMIT))),"KILL_ON_JOB_CLOSE"); key=new RSACryptoServiceProvider(2048);key.PersistKeyInCsp=false; }
  catch { Dispose();throw; }
 }
 public string Modulus {get{return Convert.ToBase64String(key.ExportParameters(false).Modulus);}}
 public string Exponent {get{return Convert.ToBase64String(key.ExportParameters(false).Exponent);}}
 public string Signature(string text) {return Convert.ToBase64String(key.SignData(Encoding.UTF8.GetBytes(text),CryptoConfig.MapNameToOID("SHA256")));}
 public bool OwnerAlive {get{try{return !owner.HasExited && owner.StartTime.ToUniversalTime().Ticks==ownerStarted;}catch{return false;}}}
 public uint Active {get{var info=new ACCOUNTING();Need(QueryInformationJobObject(job,1,ref info,(uint)Marshal.SizeOf(typeof(ACCOUNTING)),IntPtr.Zero),"Query owned active processes");return info.active;}}
 public uint ExitCode {get{return exitCode;}}
 public static long Query(string name) {
  IntPtr h=OpenJobObject(4,false,name);
  if(h==IntPtr.Zero){int e=Marshal.GetLastWin32Error();if(e==2)return -1;throw new InvalidOperationException("Owned job query failed: "+e);}
  try{var info=new ACCOUNTING();Need(QueryInformationJobObject(h,1,ref info,(uint)Marshal.SizeOf(typeof(ACCOUNTING)),IntPtr.Zero),"Query owned kernel job");return info.active;}
  finally{CloseHandle(h);}
 }
 // Windows command line quoting, including trailing backslashes before the closing quote.
 static string Quote(string s) {var b=new StringBuilder("\"");int slashes=0;foreach(char c in s){if(c=='\\'){slashes++;continue;}if(c=='\"'){b.Append('\\',slashes*2+1);b.Append(c);}else{b.Append('\\',slashes);b.Append(c);}slashes=0;}b.Append('\\',slashes*2);b.Append('"');return b.ToString();}
 public void Start(string exe,string[] args,string environment,string cwd,string log) {
  if(child!=IntPtr.Zero || stopping) throw new InvalidOperationException("Container starts once");
  var security=new SECURITY();security.length=Marshal.SizeOf(typeof(SECURITY));security.inherit=1;
  IntPtr output=CreateFile(log,0x40000000,1,ref security,1,0,IntPtr.Zero),input=IntPtr.Zero,env=IntPtr.Zero;
  Need(output!=new IntPtr(-1),"Create owned log");
  PROCESS p=new PROCESS();
  try {
   input=CreateFile("NUL",0x80000000,1|2,ref security,3,0,IntPtr.Zero);Need(input!=new IntPtr(-1),"Create input");
   var si=new STARTUP();si.cb=Marshal.SizeOf(typeof(STARTUP));si.flags=0x100;si.input=input;si.output=output;si.error=output;
   var cmd=new StringBuilder(Quote(exe));foreach(string a in args)cmd.Append(" ").Append(Quote(a));
   env=Marshal.StringToHGlobalUni(environment);
   // No BREAKAWAY flag and no breakaway job limits. Nested assignment failure never runs the child.
   Need(CreateProcess(exe,cmd,IntPtr.Zero,IntPtr.Zero,true,0x4|0x400|0x08000000,env,cwd,ref si,out p),"Create suspended child");
   child=p.process;ChildPid=p.pid;
   if(!AssignProcessToJobObject(job,child)) {int error=Marshal.GetLastWin32Error();Need(TerminateProcess(child,1),"Terminate unassigned suspended child");WaitForSingleObject(child,0xffffffff);throw new InvalidOperationException("Nested assignment refused before exec: "+error);}
   Need(ResumeThread(p.thread)!=0xffffffff,"Resume owned child");
  } catch { if(child!=IntPtr.Zero && WaitForSingleObject(child,0)!=0) {TerminateProcess(child,1);WaitForSingleObject(child,0xffffffff);}throw; }
  finally {if(p.thread!=IntPtr.Zero)CloseHandle(p.thread);if(env!=IntPtr.Zero)Marshal.FreeHGlobal(env);if(input!=IntPtr.Zero&&input!=new IntPtr(-1))CloseHandle(input);CloseHandle(output);}
 }
 public void Stop() {if(!stopping){Need(TerminateJobObject(job,1),"Terminate owned job");stopping=true;}}
 public bool Poll() {
  if(child!=IntPtr.Zero && WaitForSingleObject(child,0)==0 && !stopping) {Need(GetExitCodeProcess(child,out exitCode),"Read child exit");Stop();}
  return Active==0 && (stopping || child!=IntPtr.Zero);
 }
 public void Dispose(){if(job!=IntPtr.Zero){CloseHandle(job);job=IntPtr.Zero;}if(child!=IntPtr.Zero){CloseHandle(child);child=IntPtr.Zero;}if(key!=null){key.Dispose();key=null;}if(owner!=null){owner.Dispose();owner=null;}}
}
'@
if($PSCmdlet.ParameterSetName -eq 'query') {
 if($QueryJob -notmatch '^Local\\SceneWorksFlux3-[0-9]+-[0-9]+-[a-f0-9]{40}$'){throw 'Only an owned queue container may be queried'}
 $active=[Flux3OwnedContainer]::Query($QueryJob)
 @{exists=($active -ge 0);activeProcesses=$active;kernelObjectAbsent=($active -eq -1);context=@{sessionId=(Get-Process -Id $PID).SessionId;ownerSid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value}}|ConvertTo-Json -Depth 4 -Compress
 return
}
$boot=Get-Content -Raw -LiteralPath $BootFile|ConvertFrom-Json
$container=$null
$readyFile=[string]$boot.readyFile
$ackFile=[string]$boot.ackFile
function AtomicJson($file,$value) {
 $temp=$file+'.'+[Guid]::NewGuid().ToString('N')+'.tmp'
 [IO.File]::WriteAllText($temp,($value|ConvertTo-Json -Depth 12 -Compress),[Text.UTF8Encoding]::new($false))
 [IO.File]::Move($temp,$file)
}
try {
 $container=[Flux3OwnedContainer]::new([string]$boot.jobName,[int]$boot.supervisorPid)
 $public=@{kty='RSA';n=$container.Modulus.TrimEnd('=').Replace('+','-').Replace('/','_');e=$container.Exponent.TrimEnd('=').Replace('+','-').Replace('/','_')}
 AtomicJson $readyFile @{publicKey=$public;jobName=$boot.jobName;nonce=$boot.identity.nonce;helperPid=$PID;killOnClose=$true;context=@{sessionId=(Get-Process -Id $PID).SessionId;ownerSid=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value}}
 $started=$false; $stopping=$false
 while($true) {
  if(-not $container.OwnerAlive -or (Test-Path -LiteralPath $boot.stopFile) -or [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() -ge [long]$boot.deadline) {$stopping=$true;$container.Stop()}
  if(-not $started -and -not $stopping -and (Test-Path -LiteralPath $boot.startFile)) {
   if(-not $container.OwnerAlive -or (Test-Path -LiteralPath $boot.stopFile)){ $container.Stop() } else {
    $start=Get-Content -Raw -LiteralPath $boot.startFile|ConvertFrom-Json
    $entries=@($start.env.psobject.Properties|Sort-Object Name|ForEach-Object {$_.Name+'='+[string]$_.Value})
    $block=($entries -join [char]0)+[char]0+[char]0
    $container.Start([string]$start.exe,[string[]]$start.args,$block,[string]$start.cwd,[string]$start.log)
    $started=$true
   }
  }
  if($container.Poll()){break}
  Start-Sleep -Milliseconds 100
 }
 # ACK is signed ONLY after the kernel's complete-job accounting says ZERO. Root exit alone is
 # never an acknowledgement. A failed query/cleanup emits no ACK; finally closes the kill-on-close job.
 if($container.Active -ne 0){throw 'Owned processes remain'}
 $payload=@{kind='flux3-container-cleanup-ack';identity=$boot.identity;jobName=$boot.jobName;activeProcesses=0;cleanupVerified=$true;exitCode=$container.ExitCode;childPid=$container.ChildPid}|ConvertTo-Json -Depth 12 -Compress
 AtomicJson $ackFile @{payload=$payload;signature=$container.Signature($payload)}
} finally {if($container){$container.Dispose()}}

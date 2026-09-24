const { createApp, ref, nextTick } = Vue;

const app = createApp({
  setup() {
    const pageStage = ref('login');

    const username = ref('');
    const password = ref('');
    const loading = ref(false);

    const selectedMode = ref('file');
    const selectedCamera = ref(0);
    const cameraOptions = ref([]);
    const cameraLoading = ref(false);
    const enteringDashboard = ref(false);

    const scoreSolid = ref(0);
    const scoreStripe = ref(0);
    const foul = ref(false);
    const isPaused = ref(false);

    const videoProgress = ref(0);
    const isDragging = ref(false);

    const videoFeedUrl = ref('/video_feed');
    const currentVideoName = ref('test1.mp4');
    const uploadingVideo = ref(false);
    const videoFileInput = ref(null);
    const gameOver = ref(false);
    const winner = ref('');
    const isRecording = ref(false);
    const recordingFilename = ref('');

    const coachInput = ref('');
    const coachLoading = ref(false);
    const coachScrollRef = ref(null);
    const coachMessages = ref([
      {
        role: 'assistant',
        content: '你好，我是你的专属桌球教练。你可以问我击球策略、母球走位、黑八规则，或者让我结合当前模式与比分给建议。'
      }
    ]);

    let pollingTimer = null;
    let gameOverShown = false;

    const refreshVideoFeedUrl = () => {
      videoFeedUrl.value = `/video_feed?t=${Date.now()}`;
    };

    const scrollCoachToBottom = async () => {
      await nextTick();
      const el = coachScrollRef.value;
      if (el) el.scrollTop = el.scrollHeight;
    };

    const fetchCurrentVideoInfo = async () => {
      try {
        const res = await axios.get('/api/video/current');
        currentVideoName.value = res.data.video_name || '未选择视频';
      } catch (err) {
        console.error(err);
      }
    };

    const triggerVideoSelect = () => {
      if (videoFileInput.value) videoFileInput.value.click();
    };

    const handleVideoFileChange = async (event) => {
      const file = event.target.files?.[0];
      if (!file) return;

      const formData = new FormData();
      formData.append('video', file);
      uploadingVideo.value = true;

      try {
        const res = await axios.post('/api/video/upload', formData, {
          headers: { 'Content-Type': 'multipart/form-data' }
        });
        currentVideoName.value = res.data.video_name || file.name;
        refreshVideoFeedUrl();
        videoProgress.value = 0;
        isPaused.value = false;
        gameOver.value = false;
        winner.value = '';
        ElementPlus.ElMessage.success('本地视频已切换');
      } catch (err) {
        ElementPlus.ElMessage.error(err?.response?.data?.msg || '本地视频切换失败');
      } finally {
        uploadingVideo.value = false;
        event.target.value = '';
      }
    };

    const handleLogin = () => {
      if (loading.value) return;
      if (!username.value.trim() || !password.value.trim()) {
        ElementPlus.ElMessage.warning('请输入用户名和密码');
        return;
      }

      loading.value = true;
      axios.post('/api/login', {
        username: username.value.trim(),
        password: password.value.trim()
      }).then(res => {
        if (res.data.ok) {
          setTimeout(() => {
            loading.value = false;
            pageStage.value = 'mode_select';
            selectedMode.value = 'file';
            fetchCurrentVideoInfo();
            scrollCoachToBottom();
          }, 260);
        }
      }).catch(err => {
        loading.value = false;
        ElementPlus.ElMessage.error(err?.response?.data?.msg || '登录失败');
      });
    };

    const releaseSource = () => axios.post('/api/source/stop').catch(e => console.error(e));

    const backToLogin = () => {
      stopPolling();
      releaseSource();
      pageStage.value = 'login';
      selectedMode.value = 'file';
      cameraOptions.value = [];
      enteringDashboard.value = false;
      isRecording.value = false;
      recordingFilename.value = '';
    };

    const refreshCameras = async (silent = false) => {
      cameraLoading.value = true;
      try {
        const res = await axios.get('/api/cameras');
        cameraOptions.value = res.data.cameras || [];
        if (cameraOptions.value.length > 0) {
          const exists = cameraOptions.value.some(c => c.value === selectedCamera.value);
          if (!exists) selectedCamera.value = cameraOptions.value[0].value;
        }
      } catch (err) {
        if (!silent) ElementPlus.ElMessage.error('获取设备列表失败');
      } finally {
        cameraLoading.value = false;
      }
    };

    const chooseCameraMode = async () => {
      selectedMode.value = 'camera';
      await refreshCameras(true);
    };

    const confirmModeAndEnter = async () => {
      if (enteringDashboard.value) return;

      enteringDashboard.value = true;
      try {
        await axios.post('/api/source/select', {
          mode: selectedMode.value,
          camera_index: selectedMode.value === 'camera' ? Number(selectedCamera.value) : 0
        });

        scoreSolid.value = 0;
        scoreStripe.value = 0;
        foul.value = false;
        isPaused.value = false;
        gameOver.value = false;
        winner.value = '';
        gameOverShown = false;
        isRecording.value = false;
        recordingFilename.value = '';

        refreshVideoFeedUrl();
        pageStage.value = 'dashboard';
        startPolling();

        ElementPlus.ElMessage.success(
          selectedMode.value === 'file' ? '已启动本地视频' : `已请求连接摄像头 ${selectedCamera.value}`
        );
      } catch (err) {
        ElementPlus.ElMessage.error(err?.response?.data?.msg || '进入识别页面失败');
      } finally {
        enteringDashboard.value = false;
      }
    };

    const backToModeSelect = () => {
      stopPolling();
      releaseSource();
      pageStage.value = 'mode_select';
      isPaused.value = false;
      foul.value = false;
      gameOver.value = false;
      winner.value = '';
      gameOverShown = false;
      isRecording.value = false;
      recordingFilename.value = '';
      scrollCoachToBottom();
    };

    const startPolling = () => {
      if (pollingTimer) return;
      fetchStatus();
      pollingTimer = setInterval(fetchStatus, 700);
    };

    const stopPolling = () => {
      if (pollingTimer) {
        clearInterval(pollingTimer);
        pollingTimer = null;
      }
    };

    const fetchStatus = () => {
      axios.get('/api/status')
        .then(res => {
          const data = res.data;
          scoreSolid.value = data.solid;
          scoreStripe.value = data.stripe;
          foul.value = data.foul;
          isPaused.value = data.is_paused;
          gameOver.value = !!data.game_over;
          winner.value = data.winner || '';
          isRecording.value = !!data.recording;
          recordingFilename.value = data.recording_filename || '';

          if (data.video_name) currentVideoName.value = data.video_name;

          if (!isDragging.value && data.total_frames > 0 && selectedMode.value === 'file') {
            videoProgress.value = (data.current_frame / data.total_frames) * 100;
          }

          if (gameOver.value) {
            if (!gameOverShown) {
              ElementPlus.ElMessage.warning('黑八入袋，比赛结束！');
              gameOverShown = true;
            }
          } else {
            gameOverShown = false;
          }
        })
        .catch(err => {
          if (err?.response?.status === 401) {
            stopPolling();
            pageStage.value = 'login';
            ElementPlus.ElMessage.error('登录状态失效，请重新登录');
          }
        });
    };

    const togglePause = () => {
      if (isPaused.value) {
        axios.post('/api/control/resume').then(() => { isPaused.value = false; });
      } else {
        axios.post('/api/control/pause').then(() => { isPaused.value = true; });
      }
    };

    const restartVideo = () => {
      if (confirm('确定要重新开始比赛并重播吗？')) {
        axios.post('/api/control/restart').then(() => {
          isPaused.value = false;
          foul.value = false;
          gameOver.value = false;
          winner.value = '';
          gameOverShown = false;
          refreshVideoFeedUrl();
        });
      }
    };

    const onProgressChange = (val) => {
      axios.post('/api/control/seek', { progress: val });
    };

    const formatTooltip = (val) => Math.floor(val) + '%';

    const adjust = (team, delta) => {
      axios.post('/api/adjust', { team, delta }).then(() => {
        if (team === 'solid') scoreSolid.value += delta;
        else scoreStripe.value += delta;
      });
    };

    const clearFoul = () => {
      axios.get('/api/clear_foul').then(() => { foul.value = false; });
    };

    const resetGame = () => {
      if (confirm('仅清空比分？')) {
        axios.get('/api/reset').then(() => {
          scoreSolid.value = 0;
          scoreStripe.value = 0;
          foul.value = false;
          gameOver.value = false;
          winner.value = '';
          gameOverShown = false;
        });
      }
    };

    const toggleRecording = async () => {
      if (selectedMode.value !== 'camera') {
        ElementPlus.ElMessage.warning('只有实时摄像头模式支持录制');
        return;
      }

      try {
        if (!isRecording.value) {
          if (!confirm('是否开始录制当前实时比赛画面？')) return;
          const res = await axios.post('/api/recording/start');
          isRecording.value = true;
          recordingFilename.value = res.data.filename || '';
          ElementPlus.ElMessage.success('已开始录制');
        } else {
          const res = await axios.post('/api/recording/stop');
          isRecording.value = false;
          const savedName = res.data.filename || recordingFilename.value || '录制文件';
          recordingFilename.value = '';
          ElementPlus.ElMessage.success(`录制已停止：${savedName}`);
        }
      } catch (err) {
        ElementPlus.ElMessage.error(err?.response?.data?.msg || '录制操作失败');
      }
    };

    const sendCoachMessage = async () => {
      const text = coachInput.value.trim();
      if (!text || coachLoading.value) return;

      coachMessages.value.push({ role: 'user', content: text });
      coachInput.value = '';
      coachLoading.value = true;
      scrollCoachToBottom();

      try {
        const res = await axios.post('/api/coach/chat', {
          message: text,
          mode: selectedMode.value,
          scoreSolid: scoreSolid.value,
          scoreStripe: scoreStripe.value,
          foul: foul.value,
          history: coachMessages.value.slice(-8)
        });

        coachMessages.value.push({
          role: 'assistant',
          content: res.data.answer || '我暂时没有想到更合适的建议。'
        });
      } catch (err) {
        coachMessages.value.push({
          role: 'assistant',
          content: err?.response?.data?.msg || '专属桌球教练暂时不可用，请检查本地 Ollama 是否已启动。'
        });
      } finally {
        coachLoading.value = false;
        scrollCoachToBottom();
      }
    };

    const askQuickQuestion = (text) => {
      coachInput.value = text;
      sendCoachMessage();
    };

    const clearCoachMessages = () => {
      coachMessages.value = [{
        role: 'assistant',
        content: '对话已清空。继续问我桌球策略、走位和规则问题吧。'
      }];
      scrollCoachToBottom();
    };

    return {
      pageStage, username, password, loading, handleLogin, backToLogin,
      selectedMode, selectedCamera, cameraOptions, cameraLoading, enteringDashboard,
      chooseCameraMode, refreshCameras, confirmModeAndEnter, backToModeSelect,
      scoreSolid, scoreStripe, foul, isPaused, togglePause, restartVideo, adjust, clearFoul, resetGame,
      videoProgress, isDragging, onProgressChange, formatTooltip,
      videoFeedUrl, currentVideoName, uploadingVideo, videoFileInput, triggerVideoSelect, handleVideoFileChange,
      gameOver, winner, isRecording, recordingFilename, toggleRecording,
      coachInput, coachLoading, coachMessages, coachScrollRef, sendCoachMessage, askQuickQuestion, clearCoachMessages
    };
  }
});

app.config.compilerOptions.delimiters = ['[[', ']]'];
app.use(ElementPlus);
app.mount('#app');

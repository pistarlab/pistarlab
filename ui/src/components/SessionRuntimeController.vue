<template>
<span>
    <span v-if="item.status && item.status == 'RUNNING'">
        <b-button class="mr-2" variant="danger" v-on:click="stopSession" size="sm"><i class="fa fa-stop"></i> Abort Session</b-button>

        <span v-if="'last_update' in runtimeStatus">
            <b-button id="recording_info" :disabled="this.commandSubmitting" variant="info" v-if="!runtimeStatus.recording" class="mr-2" v-on:click="sessionCommand('recording','enabled',true)" size="sm">
                <i class="fa fa-video"></i> Start Recording</b-button>
            <b-button :disabled="this.commandSubmitting" variant="warning" v-if="runtimeStatus.recording" class="mr-2" v-on:click="sessionCommand('recording','enabled',false)" size="sm">
                <i class="fa fa-stop"></i> Stop Recording</b-button>

            <b-popover target="recording_info" triggers="hover" placement="right">
                <template #title>Start Recording </template>
                <span><span>Click to start recording immediately. Note: This session will automatically record every </span>{{JSON.parse(item.config).episode_record_freq}} episodes</span>.

            </b-popover>

            <b-button :disabled="this.commandSubmitting" v-if="!runtimeStatus.runtime_logging" class="mr-2" v-on:click="sessionCommand('runtime_logging','enabled',true)" size="sm">
                <i class="fa fas-monitor-heart-rate"></i> Enable Logging</b-button>
            <b-button :disabled="this.commandSubmitting" variant="warning" v-if="runtimeStatus.runtime_logging" class="mr-2" v-on:click="sessionCommand('runtime_logging','enabled',false)" size="sm">
                <i class="fa fa-stop"></i> Disable Logging</b-button>

            <span>
                <span class="mr-2">Step Per Second</span>
                <b-form-select style="width:100px" :disabled="this.commandSubmitting" @change="updateStepSpeed()" v-model="stepSpeed" :options="stepSpeedOptions" size="sm"></b-form-select>
            </span>

        </span>
    </span>

</span>
</template>

<script>
import axios from "axios";

export default {
    props: {
        item: Object,

    },
    data() {
        return {
            playingEpisode: false,
            playingLive: false,
            runtimeStatus: {},
            stepSpeed: null,
            commandSubmitting: false,
            error: null,
            message: null,
            timer: null,
            stepSpeedOptions: [{
                    value: null,
                    text: 'Unlimited'
                },
                {
                    value: 2,
                    text: '2'
                },
                {
                    value: 5,
                    text: '5'
                },
                {
                    value: 15,
                    text: '15'
                },
                {
                    value: 60,
                    text: '60'
                },
                {
                    value: 120,
                    text: '120'
                },
                {
                    value: 180,
                    text: '180'
                }
            ],

        };
    },
    mounted() {
        //
    },
    methods: {
        stopSession() {
            if (this.item) {
                axios
                    .get(
                        `${this.appConfig.API_URL}/api/admin/task/stop/${this.item.task.ident}`
                    )
                    .then((response) => {
                        let message = response.data["message"];
                        this.makeToast(message, "User Abort Request", "info")
                        console.log(` ${message}`)
                        this.refreshData()
                    })
                    .catch((e) => {
                        this.error = e;
                        this.message = this.error;
                    });
            }
        },
        updateStepSpeed() {
            this.sessionCommand("step_speed_limitor", "value", this.stepSpeed)
        },
        sessionCommand(command, param, value) {
            if (this.item && this.running) {
                console.log(`Session Command ${command} ${param} ${value}`)
                this.commandSubmitting = true
                let params = {}
                params[param] = value
                console.log(params)
                axios
                    .post(
                        `${this.appConfig.API_URL}/api/session_command/${this.item.ident}/${command}`,
                        params
                    )
                    .then((response) => {
                        console.log("HI")
                        if (response.data.last_update != undefined) {
                            this.updateSessionStatus(response.data)
                        } else {
                            console.log("Command update response not received")
                        }
                        this.refreshData()
                        this.commandSubmitting = false

                    })
                    .catch((e) => {
                        console.log(e)
                        this.error = e;
                        this.message = this.error;
                        this.commandSubmitting = false

                    });
            }
        },
        updateSessionStatus(data) {
            this.runtimeStatus = data
            this.stepSpeed = this.runtimeStatus.step_speed_limitor
            let message = JSON.stringify(data, null, 2);
            console.log(` ${message}`)
        },
        getSessionStatus() {
            console.log("Getting status")
            if (this.item && this.running && !this.commandSubmitting) {
                axios
                    .get(
                        `${this.appConfig.API_URL}/api/session_runtime_status/${this.item.ident}`
                    )
                    .then((response) => {
                        this.updateSessionStatus(response.data)

                    })
                    .catch((e) => {
                        this.error = e;
                        this.message = this.error;
                        this.makeToast(this.message, "Error getting status", "info")
                    });
            }
        },
        refreshData() {
            this.getSessionStatus()
        }

    },
    computed: {
        //
        running() {
            return this.item.status == "RUNNING"
        }

    },
    // Fetches posts when the component is created.
    created() {
        //
        this.timer = setInterval(this.refreshData, 2000);

    },
    beforeDestroy() {
        clearInterval(this.timer);

    }
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->

<style scoped lang="scss">

</style>

<template>
<div>
    <b-alert v-if="error" variant="danger">{{error}}</b-alert>
    <b-form-group>
        <b-container fluid>
            <b-row>

                <b-col cols=1>
                </b-col>

                <b-col cols=4 class="h4">
                    Snapshot Info
                </b-col>

                <b-col>
                    <div class="h4">Environment Stats: </div>
                    <b-row class="h6 mb-0 pb-0">
                        <b-col cols=4>
                            Environment Spec ID
                        </b-col>

                        <b-col>
                            Steps
                        </b-col>
                        <b-col>
                            Episodes
                        </b-col>
                        <b-col>
                            Reward
                        </b-col>
                        <b-col>
                            Sessions
                        </b-col>
                    </b-row>
                </b-col>
            </b-row>
            <hr />

            <div v-if="loading">
                Loading...
            </div>
            <div v-else-if="snapshots==null || Object.keys(snapshots).length ==0">
                No snapshots found
            </div>
            <div v-for="(item,idx) in snapshots" v-bind:key="idx">
                <b-row>
                    <b-col cols=1 class="text-center">
                        <b-form-radio v-model="selected" @change="select(item.snapshot_id)" name="xxx" :value="item.snapshot_id"></b-form-radio>
                    </b-col>
                    <b-col>
                        <span class="h4">{{item.id}} - {{item.seed}} - {{item.snapshot_version}}</span>
                    </b-col>

                </b-row>

                <b-row>
                    <b-col cols=1>

                    </b-col>
                    <b-col cols=4>
                        <div v-if="!online">
                            <span v-if="item.published"><i class="fa fa-cloud"></i> Published</span>
                            <span v-else><i class="fa fa-hdd"></i> Not Published</span>
                        </div>
                        <div>
                            <span>Created by: {{item.submitter_id}}</span>
                        </div>
                        <div>
                            <span>{{item.snapshot_description}}</span>
                        </div>
                    </b-col>

                    <b-col class="">

                        <div v-if="item.env_stats == null|| Object.keys(item.env_stats).length ==0">
                            No Stats Found
                        </div>
                        <div v-else>

                            <div v-for="(stats,k) in item.env_stats" v-bind:key="k">
                                <b-row>
                                    <b-col cols=4>
                                        {{k}}
                                    </b-col>

                                    <b-col>
                                        {{stats['step_count']}}
                                    </b-col>
                                    <b-col>
                                        {{stats['episode_count']}}
                                    </b-col>
                                    <b-col>
                                        {{stats['best_ep_reward_total']}}
                                    </b-col>
                                    <b-col>
                                        {{stats['session_count']}}
                                    </b-col>
                                </b-row>
                            </div>
                        </div>
                    </b-col>
                </b-row>

                <!-- <b-col>

                        <div>
                            <span class="data_label mt-1">Observation Space: </span>
                            <span>
                                <SpaceInfo :space="item.config.interfaces.run.observation_space"></SpaceInfo>
                            </span>
                        </div>
                    </b-col>
                    <b-col class="">

                        <div>
                            <span class="data_label mt-1">Action Space: </span>
                            <span>
                                <SpaceInfo :space="item.config.interfaces.run.action_space"></SpaceInfo>
                            </span>
                        </div>
                    </b-col> -->
                <!-- <b-row class="mt-2">
                    <b-col class="">
                                                <div>
                            <span class="data_label mt-1">Submitter Id: </span>
                            <span>{{item.submitter_id}}</span>

                        </div>
                                                <div>
                            <span class="data_label mt-1">Version Id: </span>
                            <span>{{item.snapshot_version}}</span>

                        </div>
                        <div>
                            <span class="data_label mt-1">Creation Time: </span>
                            <span>{{item.creation_time}}</span>

                        </div>
                        <div>
                            <span class="data_label mt-1">Observation Space </span>
                            <pre>{{JSON.stringify(item.config.observation_space,null,2)}}</pre>
                        </div>
                    </b-col>
                    <b-col class="">
                        <div>
                            <span class="data_label mt-1">Action Space </span>
                            <pre>{{JSON.stringify(item.config.action_space,null,2)}}</pre>
                        </div>
                    </b-col>
                </b-row> -->
                <hr />

            </div>

        </b-container>
    </b-form-group>
</div>
</template>

<script>
import axios from "axios";
import {
    appConfig
} from "../app.config";
import {
    timedelta,
    timepretty
} from "../funcs";

import SpaceInfo from "./SpaceInfo.vue"

export default {
    props: {
        specId: String,
        online: {
            'type': Boolean,
            'default': false
        }
    },
    components: {
        // SpaceInfo
    },

    data() {
        return {
            snapshots: [],
            error: null,
            selected: null,
            loading: true

        };
    },
    mounted() {
        //
    },
    methods: {
        loadLocalData() {
            console.log("Loading local snapshots")
            this.error = null
            this.loading = true
            axios
                .get(`${appConfig.API_URL}/api/snapshots/list/${this.specId}`)
                .then((response) => {
                    this.snapshots = response.data["items"]
                    this.loading = false
                })
                .catch((e) => {
                    this.error = e;
                    this.loading = false
                });
        },
        loadOnlineData() {
            console.log("Loading online snapshots")
            this.error = null
            this.loading = true
            axios
                .get(`${appConfig.API_URL}/api/snapshots/public/list/spec_id/${this.specId}`)
                .then((response) => {
                    this.snapshots = response.data["items"]
                    this.loading = false

                })
                .catch((e) => {
                    this.error = e;
                    this.loading = false
                });
        },
        loadData() {
            if (this.online == true) {
                this.loadOnlineData()
            } else {
                this.loadLocalData()
            }

        },
        select(uid) {
            console.log(uid)
            this.$emit('input', uid)

        },

    },
    computed: {

    },
    // Fetches posts when the component is created.
    created() {
        this.loadData();

    },
    beforeDestroy() {
        //

    }
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->

<style scoped lang="scss">

</style>

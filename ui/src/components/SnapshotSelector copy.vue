<template>
<div>
    <b-alert v-if="error" variant="danger">{{error}}</b-alert>
    <b-container fluid>

        <div v-for="(item,idx) in snapshots" v-bind:key="idx">
            <b-row>
                <b-col>
                    <div>{{item.snapshot_id}}</div>
                </b-col>
                <b-col>
                    <b-button size="sm" variant="primary" @click="select(item.snapshot_id)">Select</b-button>

                </b-col>
            </b-row>
            <div class="mt-2"></div>
            <b-row>
                <b-col class="">
                    <div>
                        <span class="data_label mt-1">Version: </span>
                        <span>{{item.snapshot_version}}</span>
                    </div>
 
                    <div>
                        <span class="data_label mt-1">Submitter Id: </span>
                        <span>{{item.user_id}}</span>

                    </div>
                </b-col>
                <b-col class="">
                    <div>
                        <span class="data_label mt-1">Creation Time: </span>
                        <span>{{item.creation_time}}</span>

                    </div>
                                        <div>
                        <span class="data_label mt-1">Description: </span>
                        <span>{{item.snapshot_description}}</span>
                    </div>

                </b-col>

            </b-row>
            <b-row class="mt-2">
                <b-col class="">
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
            </b-row>
            <hr />

        </div>
    </b-container>
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

export default {
    props: {
        specId: String,
    },

    data() {
        return {
            snapshots: [],
            error:null

        };
    },
    mounted() {
        //
    },
    methods: {
        loadData() {
            this.error=null
            axios
                .get(`${appConfig.API_URL}/api/snapshots/list/${this.specId}`)
                .then((response) => {
                    this.snapshots = response.data["items"]

                })
                .catch((e) => {
                    this.error = e;
                });
        },
        select(uid) {
            this.$emit('click', uid)

        },

    },
    computed: {
        //

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

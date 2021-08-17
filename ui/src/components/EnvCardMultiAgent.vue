<template>
<div v-if="item.ident">

    <b-card no-body class="overflow-hidden card-shadow">
        <b-row no-gutters>
            <b-col md="4">
                <b-card-img :src="`${appConfig.API_URL}/api/env_preview_image/${item.ident}`" alt="" style="max-height:350px; max-width:400px; min-height:200px; min-width:250px; height:auto; width: auto;"></b-card-img>
            </b-col>
            <b-col md="8">
                <b-card-body :title="item.ident" >
                    <b-card-text>
                        <div class="mt-auto">

                            <div class="data_label">Environment Group: {{ item.environment.ident }}</div>
                            <div class="data_label">Observation Spaces:
                                > 
                                <div for="observation in ">
                                    {{meta.observation_spaces}})
                                </div>

                            </div>
                            <div class="data_label">Action Space:
                                <span v-if="meta && meta.action_space">
                                    {{meta.action_space.class_name}} (args={{meta.action_space.args}},{{meta.action_space.kargs}})
                                </span>
                                <span v-else>Undefined</span>
                            </div>

                        </div>
                    </b-card-text>
                </b-card-body>
            </b-col>
        </b-row>
    </b-card>

</div>
</template>

<script >
import {
    appConfig
} from "../app.config";
export default {
    props: {
        item: Object
    },
    data() {
        return {
            appConfig

        };
    },
    mounted() {
        //
    },
    methods: {
        //

    },
    computed: {

        meta(){
            return JSON.parse(this.item.meta)
        }
        //

    },
    // Fetches posts when the component is created.
    created() {
        //

    },
    beforeDestroy() {
        //

    }
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->

<style>

</style>
